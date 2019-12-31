# -*- coding: utf-8 -*-
import os
import sys
import logging
import codecs
import argparse
from imp import reload
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import multiprocessing
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xlwings as xw
from jinja2 import Template
from sklearn.model_selection import train_test_split, KFold
try:
    import xgboost as xgb
except:
    pass

import utils3.plotting as pl
import utils3.metrics as mt
import utils3.misc_utils as mu
import utils3.feature_selection as fs
from utils3.data_io_utils import *


class XGBModel(object):
    def __init__(self, var_dict, DATA_PATH, RESULT_PATH, model_label_version, y_train, y_test=None,
                params_selection_method=['XGBExp'], nfold=5):
        """
        params_selection_method (list): ['XGBExp', 'XGBGrid', 'XGBRandom']
        """
        self.var_dict = var_dict
        self.DATA_PATH = DATA_PATH
        self.RESULT_PATH = RESULT_PATH
        xgboost_data_spec_dict = load_data_from_pickle(DATA_PATH, '%s_XGBoost输出数据和分箱明细.pkl' % model_label_version)
        self.X_train_xgboost = xgboost_data_spec_dict['X_train_xgboost']
        self.X_test_xgboost = xgboost_data_spec_dict.get('X_test_xgboost', '')
        self.y_train = y_train
        self.y_test = y_test
        self.params_selection_method = params_selection_method
        self.NFOLD = nfold
        self.model_label_version = model_label_version

    def run(self, params_range=None, param_experience=None, xgb_select=True,space=None, isPrint=True, fit_params=None):
        if xgb_select:
            self.xgb_select()
        else:
            self.xgb_select(auto=False)
            
        self.get_params(params_range, param_experience, space, isPrint, fit_params)

        if 'XGBExp' in self.params_selection_method:
            logging.info("XGBoost模型使用经验参数建模")
            self.fit_model(self.param_experience,'%sXGBExp' % self.model_label_version, self.NFOLD)

        if 'XGBGrid' in self.params_selection_method:
            logging.info("XGBoost模型使用GridSearch调参所选最优参数建模")
            self.fit_model(self.gridsearch_params,'%sXGBGrid' % self.model_label_version, self.NFOLD)

        if 'XGBRandom' in self.params_selection_method:
            logging.info("XGBoost模型使用RandomSearch调参所选最优参数建模")
            self.fit_model(self.randomsearch_params, '%sXGBRandom' % self.model_label_version, self.NFOLD)

        if 'XGBHyperopt' in self.params_selection_method:
            logging.info("XGBoost模型使用Hyperopt调参所选最优参数建模")
            self.fit_model(self.hyperopt_params, '%sXGBHyperopt' % self.model_label_version, self.NFOLD)
            
    def fit_model(self, param, model_label, nfold=5):
        if len(self.X_test_xgboost) > 0:
            all_X = pd.concat([self.X_train_xgboost, self.X_test_xgboost])[self.selected]
            all_y = pd.concat([self.y_train, self.y_test])
        else:
            all_X = self.X_train_xgboost[self.selected].copy()
            all_y = self.y_train

        logging.info("XGB模型在参数已调整好后，使用全部TRAIN&TEST数据集建模")
        all_p, xgbmodel_result, importance_df = self.xgboost(all_X, all_y, param, sample_weight=sample_weight)

        if len(self.X_test_xgboost) > 0:
            p_train = all_p.loc[self.X_train_xgboost.index].copy()
            p_test = all_p.loc[self.X_test_xgboost.index].copy()
        else:
            p_train = all_p
            p_test = p_train

        final_cols = xgbmodel_result.__dict__['feature_names']
        all_p_cv = self.xgboost_cv(all_X[final_cols], all_y, param, nfold, sample_weight=sample_weight)

        model_result = {
            'model_final': xgbmodel_result,
            'p_train': p_train,
            'p_test': p_test,
            'all_p_cv': all_p_cv,
            'nfold': nfold,
            'model_label': model_label
        }
        save_data_to_pickle(model_result, self.DATA_PATH, '%s模型结果.pkl' % model_label)
        importance_df.to_excel(os.path.join(self.RESULT_PATH, '%s模型变量重要性排序.xlsx' % model_label), index=False)
        logging.info("""
        XGB 模型建模输出：
        1. 模型文件，预测概率等存储于： %s
        2. 模型变量重要性排序存储于：%s
        """ % (os.path.join(self.DATA_PATH, '%s模型结果.pkl' % model_label),
               os.path.join(self.RESULT_PATH, '%s模型变量重要性排序.xlsx' % model_label))
        )
        
    def xgb_select(self,auto=True):
        def _obtain_select(imp):
            # TODO verify how rand1 and rand2 work in model
            try:
                imp_threshold1 = imp.loc[imp.var_code=='rand1', 'xgboost_rank'].fillna(0).iloc[0]
            except:
                imp_threshold1 = 0
            try:
                imp_threshold2 = imp.loc[imp.var_code=='rand2', 'xgboost_rank'].fillna(0).iloc[0]
            except:
                imp_threshold2 = 0
            imp_threshold = max(imp_threshold1, imp_threshold2)
            selected2 = imp.loc[imp.xgboost_rank<=imp_threshold, 'var_code'].tolist()
            selected = [x for x in selected2 if x not in ['rand1','rand2']]
            top50pct = imp.loc[~imp.var_code.isin(['rand1', 'rand2'])].iloc[:int((len(imp)-2)/2)].var_code.tolist()
            top50 = imp.loc[~imp.var_code.isin(['rand1', 'rand2'])].iloc[:50].var_code.tolist()
            top50_max = max(len(top50pct), len(top50))
            if len(selected) < top50_max:
                if len(top50pct) >= len(top50):
                    return top50pct
                else:
                    return top50

            return selected

        self.X_train_xgboost['rand1'] = np.random.random(len(self.X_train_xgboost))
        self.X_train_xgboost['rand2'] = np.random.random(len(self.X_train_xgboost))
        train_xgb_rank = fs.FeatureSelection().xgboost(self.X_train_xgboost, self.y_train)
        selected = _obtain_select(train_xgb_rank)

        if len(self.X_test_xgboost) > 10000:
            test_xgb_rank = fs.FeatureSelection().xgboost(self.X_test_xgboost, self.y_test)
            #jzw modify 2018.12.05
            test_selected = _obtain_select(test_xgb_rank)
            selected = list(set(selected).intersection(test_selected))

        self.selected = selected
        if auto:
            cols_filter = load_data_from_json(self.RESULT_PATH, 'variable_filter.json')
            cols_filter['xgboost_selected'] = selected
            save_data_to_json(cols_filter, self.RESULT_PATH, 'variable_filter.json')
        train_xgb_rank.to_excel(os.path.join(self.RESULT_PATH, 'xgb_select_rank.xlsx'), index=False)

        logging.info("XGB建模经排序选择变量更新于%s" % os.path.join(self.RESULT_PATH, 'variable_filter.json'))
        logging.info("XGB建模排序选择排序存储于%s" % os.path.join(self.RESULT_PATH, 'xgb_select_rank.xlsx'))
    
    def get_params(self, params_range=None, param_experience=None,space=None, isPrint=True, fit_params=None):
        # logging.info('param_range: %s' % params_range)
        # 经验
        if 'XGBExp' in self.params_selection_method:
            if param_experience == None:
                self.param_experience = {
                    'max_depth': 2,
                    'min_samples_leaf ': int(len(self.X_train_xgboost)*0.02),
                     'eta': 0.1,
                     'objective': 'binary:logistic',
                     'subsample': 0.8,
                     'colsample_bytree': 0.8,
                     'gamma': 0,
                     'silent': 0,
                     'eval_metric':'auc'
                }
            else:
                self.param_experience = param_experience

            self.param_experience = fs.ParamsTuning()\
                    .xgboost_tree(self.X_train_xgboost[self.selected], self.y_train,self.param_experience)
            save_data_to_json(self.param_experience, self.DATA_PATH, 'xgboost_params_experience.json')
            logging.info("经验参数存储于%s" % os.path.join(self.DATA_PATH, 'xgboost_params_experience.json'))

        if len(self.X_test_xgboost) > 0:
            if 'XGBGrid' in self.params_selection_method:
                logging.info('开始网格调参')
                self.gridsearch_params = fs.ParamsTuning()\
                        .xgboost_gridsearch(self.X_train_xgboost[self.selected], self.y_train,
                                            self.X_test_xgboost[self.selected], self.y_test,
                                            self.NFOLD, params_range)

                self.gridsearch_params = fs.ParamsTuning()\
                        .xgboost_tree(self.X_train_xgboost[self.selected], self.y_train,
                                      self.gridsearch_params)
                logging.info('网格调参完成！')
                save_data_to_json(self.gridsearch_params, self.DATA_PATH, 'xgboost_params_gridsearch.json')
                logging.info("网格调参最优参数存储于%s" % os.path.join(self.DATA_PATH, 'xgboost_params_gridsearch.json'))

            if 'XGBRandom' in self.params_selection_method:
                logging.info("开始随机调参")
                self.randomsearch_params = fs.ParamsTuning()\
                        .xgboost_randomgridsearch(self.X_train_xgboost[self.selected], self.y_train,
                                                  self.X_test_xgboost[self.selected], self.y_test,
                                                  self.NFOLD, params_range)

                self.randomsearch_params = fs.ParamsTuning()\
                        .xgboost_tree(self.X_train_xgboost[self.selected], self.y_train,
                                      self.randomsearch_params)
                logging.info('随机调参完成！')
                # save_data_to_json(self.randomsearch_params, self.DATA_PATH, 'xgboost_params_randomsearch.json')
                logging.info("随机调参最优参数存储于%s" % os.path.join(self.DATA_PATH, 'xgboost_params_randomsearch.json'))

            if 'XGBHyperopt' in self.params_selection_method:
                logging.info("开始超参调参")
                self.hyperopt_params = fs.ParamsTuning()\
                        .xgboost_hyperopt(self.X_train_xgboost[self.selected], self.y_train,
                                                  self.X_test_xgboost[self.selected], self.y_test,
                                                  space, isPrint, fit_params)
                # logging.info('hyperopt_params==%s, space==%s' % (self.hyperopt_params,space))
                self.hyperopt_params = fs.ParamsTuning()\
                        .xgboost_tree(self.X_train_xgboost[self.selected], self.y_train,
                                      self.hyperopt_params)
                logging.info('随机调参完成！')
                # save_data_to_json(self.hyperopt_params, self.DATA_PATH, 'xgboost_params_hyperopt.json')
                logging.info("随机调参最优参数存储于%s" % os.path.join(self.DATA_PATH, 'xgboost_params_hyperopt.json'))

        else:
            X_train, X_test, y_train, y_test = train_test_split(self.X_train_xgboost[self.selected], self.y_train, test_size=0.3, random_state=0)

            if 'XGBGrid' in self.params_selection_method:
                logging.info('开始网格调参')
                self.gridsearch_params = fs.ParamsTuning()\
                        .xgboost_gridsearch(X_train, y_train, X_test, y_test,
                                            self.NFOLD, params_range)

                self.gridsearch_params = fs.ParamsTuning()\
                        .xgboost_tree(X_train, y_train, self.gridsearch_params)
                logging.info('网格调参完成！')
                save_data_to_json(self.gridsearch_params, self.DATA_PATH, 'xgboost_params_gridsearch.json')
                logging.info("网格调参最优参数存储于%s" % os.path.join(self.DATA_PATH, 'xgboost_params_gridsearch.json'))

            if 'XGBRandom' in self.params_selection_method:
                logging.info("开始随机调参")
                self.randomsearch_params = fs.ParamsTuning()\
                        .xgboost_randomgridsearch(X_train, y_train, X_test, y_test,
                                            self.NFOLD, params_range)

                self.randomsearch_params = fs.ParamsTuning()\
                        .xgboost_tree(X_train, y_train, self.randomsearch_params)
                logging.info('随机调参完成！')
                save_data_to_json(self.randomsearch_params, self.DATA_PATH, 'xgboost_params_randomsearch.json')
                logging.info("随机调参最优参数存储于%s" % os.path.join(self.DATA_PATH, 'xgboost_params_randomsearch.json'))

            if 'XGBHyperopt' in self.params_selection_method:
                logging.info("开始超参调参")
                self.hyperopt_params = fs.ParamsTuning()\
                        .xgboost_hyperopt(self.X_train_xgboost[self.selected], self.y_train,
                                                  self.X_test_xgboost[self.selected], self.y_test,
                                                  space, isPrint, fit_params)
                self.hyperopt_params = fs.ParamsTuning()\
                        .xgboost_tree(self.X_train_xgboost[self.selected], self.y_train,
                                      self.hyperopt_params)
                logging.info('随机调参完成！')
                save_data_to_json(self.hyperopt_params, self.DATA_PATH, 'xgboost_params_hyperopt.json')
                logging.info("随机调参最优参数存储于%s" % os.path.join(self.DATA_PATH, 'xgboost_params_hyperopt.json'))
                
    def xgboost(self, X, y, param):
        """
        training and prediction
        params 应该已经调参完成了
        """
        logging.info('param==%s' % param)
        dtrain = xgb.DMatrix(X, label=y)
        xgbmodel = xgb.train(param, dtrain, num_boost_round=param['num_boost_round'])
        p_train = xgbmodel.predict(dtrain)
        p_train = pd.Series(p_train, index=X.index)

        importance = xgbmodel.get_fscore()
        importance_df = pd.DataFrame.from_dict(importance, "index")
        importance_df = importance_df.reset_index().rename(columns={"index": "feature", 0: "fscore"})
        importance_df["imp_pct"] = importance_df["fscore"] / importance_df["fscore"].sum()
        importance_df = importance_df.sort_values("imp_pct", ascending=False)
        importance_df.reset_index(drop=True, inplace=True)

        if importance_df.fscore.isnull().sum() > 0:
            na_cols = importance_df.loc[importance_df.fscore.isnull(), 'feature'].tolist()
            logging.info("XGBoost建模用已调好参数fit。importance score为空的数量为：%s, out of %s. %s" % (len(na_cols), len(importance_df), json.dumps(na_cols)))
            return self.xgboost(X.drop(na_cols, 1), y, param)
        else:
            return p_train, xgbmodel, importance_df

    def xgboost_cv(self, X, y, param, folds=5):
        kf = KFold(n_splits=folds, shuffle=True)
        p_test_list = []
        for train_index , test_index in kf.split(X):
            _, xgbmodel, _ = self.xgboost(X.iloc[train_index], y.iloc[train_index],param)
            dtest = xgb.DMatrix(X.iloc[test_index], label=y.iloc[test_index])
            p_test = xgbmodel.predict(dtest)
            p_test = pd.Series(p_test, index=y.iloc[test_index].index)
            p_test_list.append(p_test)
        all_p_test = pd.concat(p_test_list)
        return all_p_test.loc[y.index]
