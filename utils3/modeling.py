# encoding utf-8
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


class LogisticModel(object):
    def __init__(self, var_dict, y_train=None, y_test=None, with_test=True):
        self.with_test = with_test
        self.y_train = y_train
        self.y_test = y_test
        self.var_dict = var_dict


    def logistic_regression(self, X_train, X_test, y_train):
        X_train = X_train.copy()
        X_train = sm.add_constant(X_train)
        X_test = X_test.copy()
        X_test = sm.add_constant(X_test)
        logit = sm.GLM(y_train, X_train, family=sm.families.Binomial())#Gamma Binomial 尝试哪个效果好
        result = logit.fit()
        p_train = result.fittedvalues
        p_test= result.predict(X_test)
        return result, p_train, p_test

    def logistic_regression_sw(self,X_train,X_test,y_train,freq_weights):
        X_train=X_train.copy()
        X_train = sm.add_constant(X_train)
        X_test=X_test.copy()
        X_test = sm.add_constant(X_test)
        logit = sm.GLM(y_train, X_train,family= sm.families.Binomial(),freq_weights=freq_weights)
        result=logit.fit()
        p_train=result.fittedvalues
        p_test= result.predict(X_test)
        return result, p_train, p_test

    def logistic_cv(self, X, y, folds=5):
        kf = KFold(n_splits=folds, shuffle=True)
        p_test_list = []
        for train_index , test_index in kf.split(X):
            model_test, _, p_test = self.logistic_regression(X.iloc[train_index], \
                                                                     X.iloc[test_index],\
                                                                     y.iloc[train_index])
            p_test = pd.Series(p_test, index=X.iloc[test_index].index)
            p_test_list.append(p_test)
        all_p_test = pd.concat(p_test_list)
        return all_p_test.loc[y.index]


    def logistic_cv_sw(self,X,y,freq_weights,folds=5):
        kf=KFold(n_splits=folds,shuffle=True)
        p_test_list=[]
        for train_index , test_index in kf.split(X):
            model_test, _, p_test = self.logistic_regression_sw(X.iloc[train_index], \
                                                                         X.iloc[test_index],\
                                                                         y.iloc[train_index],freq_weights[train_index])
            p_test = pd.Series(p_test, index=X.iloc[test_index].index)
            p_test_list.append(p_test)
        all_p_test = pd.concat(p_test_list)
        return all_p_test.loc[y.index]

    def model_stat_result(self, model_final, var_dict):
        '''
        模型的统计
        Args:
        model_final: 模型的最终fit结果(目前只针对logstic模型)
        var_dict(pd.DataFrame()): 变量字典

        Returns:
        model_stat_result(pd.DataFrame()):含有中文字段解释的统计结果
        '''
        def f(value):
            return '%.3f' % value

        var_CN = []
        for i in list(model_final.params.index):
            if i=='const':
                var_CN.append('截距')
            else:
                var_CN.append(var_dict.loc[var_dict['指标英文']==i,'指标中文'].item())
        model_stat_result = pd.DataFrame(model_final.params)
        model_stat_result['指标中文'] = var_CN
        model_stat_result = model_stat_result.rename(columns={0:'coef'})
        model_stat_result['std err'] = model_final.bse
        model_stat_result['Chi-Square'] = (model_stat_result['coef']/model_stat_result['std err'])**2
        model_stat_result['Pr>ChiSq'] = model_final.pvalues
        model_stat_result['[0.025'] = (model_stat_result['coef'] - 1.96*model_stat_result['std err'])
        model_stat_result['0.975]'] = (model_stat_result['coef'] + 1.96*model_stat_result['std err'])
        for i in model_stat_result.columns:
            try:
                model_stat_result[i] = model_stat_result[i].apply(f)
            except:
                pass
        return model_stat_result[['指标中文', 'coef', 'std err', 'Chi-Square', 'Pr>ChiSq', '[0.025', '0.975]']]



    def fit_model(self, model_label, X_train_transformed, X_test_transformed, \
                  in_model, nfold=5):
        """
        建模并将系数，模型，预测的概率等结果统一返回
        Args:
        model_label(str): 同一个建模项目下面不同版本的标号
        X_train_transformed (pd.DataFrame): woe转换好的数据，用来建模
        X_test_transformed (pd.DataFrame): woe转换好的数据，用来建模
        in_model (list): list of columns names (aka variable names) to be included
            in the model
        nfold (int): default=5

        Returns:
        model_result (dict): 建模结果
        """
        if sum(X_train_transformed.index == self.y_train.index) != len(self.y_train):
            print('X_train_transformed and y_train index not matched, common number: %s' \
                    % sum(X_train_transformed.index == self.y_train.index))
            return None

        model_final, p_train, p_test = self.logistic_regression(X_train_transformed[in_model],\
                                                               X_test_transformed[in_model],\
                                                               self.y_train)
        self.train_stat = self.model_stat_result(model_final, self.var_dict)

        if self.with_test == True:
            model_final_test, _, _ = self.logistic_regression(X_test_transformed[in_model],\
                                                            X_test_transformed[in_model],\
                                                            self.y_test)
            self.test_stat = self.model_stat_result(model_final_test, self.var_dict)

            X_transformed_all = pd.concat([X_train_transformed[in_model], X_test_transformed[in_model]])
            y_all = pd.concat([self.y_train, self.y_test])
        else:
            test_stat = 'No test set'
            X_transformed_all = X_train_transformed[in_model]
            y_all = self.y_train
            self.test_stat = None

        all_p_cv = self.logistic_cv(X_transformed_all, y_all, folds=nfold)

        self.model_result = {
            'model_final': model_final,
            'p_train': p_train,
            'p_test': p_test,
            'train_coef': self.train_stat,
            'test_coef': self.test_stat,
            'all_p_cv': all_p_cv,
            'nfold': nfold,
            'model_label': model_label
        }
        self.model_label = model_label

        return self.model_result


    def fit_model_sw(self,model_label,X_train_transformed,X_test_transformed,sample_weight_train,sample_weight_test,in_model,nfold=5):
        """
        含有sample_weight建模，返回系数、模型、预测的概率的结果
        Args:
        model_label(str): 同一个建模项目下面不同版本的标号
        X_train_transformed (pd.DataFrame): woe转换好的数据，用来建模
        X_test_transformed (pd.DataFrame): woe转换好的数据，用来建模
        sample_weight(Series):样本权重
        in_model (list): list of columns names (aka variable names) to be included
        nfold (int): default=5

        Retrun:
        model_result(dict):建模结果
        """
        if sum(X_train_transformed.index == self.y_train.index) != len(self.y_train):
            print('X_train_transformed and y_train index not matched, common number: %s' \
                    % sum(X_train_transformed.index == self.y_train.index))
            return None

        model_final, p_train, p_test = self.logistic_regression_sw(X_train_transformed[in_model],\
                                                                X_test_transformed[in_model],self.y_train,sample_weight_train)
        print('good_sw')
        self.train_stat = self.model_stat_result(model_final, self.var_dict)
        if self.with_test == True:
            model_final_test, _, _ = self.logistic_regression_sw(X_test_transformed[in_model],\
                                                                    X_test_transformed[in_model],self.y_test,sample_weight_test)
            self.test_stat = self.model_stat_result(model_final_test, self.var_dict)
            X_transformed_all = pd.concat([X_train_transformed[in_model], X_test_transformed[in_model]])
            y_all = pd.concat([self.y_train, self.y_test])
            sample_weight=pd.concat([sample_weight_train,sample_weight_test])
        else:
            test_stat = 'No test set'
            X_transformed_all = X_train_transformed[in_model]
            y_all = self.y_train
            sample_weight=sample_weight_train
            self.test_stat = None
        all_p_cv = self.logistic_cv_sw(X_transformed_all, y_all,sample_weight,folds=nfold)
        self.model_result={
        'model_final': model_final,
        'p_train': p_train,
        'p_test': p_test,
        'train_coef': self.train_stat,
        'test_coef': self.test_stat,
        'all_p_cv': all_p_cv,
        'nfold': nfold,
        'model_label': model_label
        }
        self.model_label = model_label
        return self.model_result


    def plot_for_the_model(self, RESULT_PATH, result_dict, with_test=True):
        """
        plot AUC， KS图 for train, test, cv

        Args:
        result_dict (dict): the dict returned by self.fit_model()。单独pass进来而不是直接用
            self.result 是为了此函数的延展性，可以plot统一个建模项目中任意一个版本的模型结果的图
            需包含key值为：'p_train', 'p_test', 'all_p_cv', 'model_label', 'nfold'
        """
        pl.ks_new(np.array(result_dict['p_train']), np.array(self.y_train), save_label='%s_TRAIN' % result_dict['model_label'], \
                  plot=True, result_path=RESULT_PATH)
        pl.print_AUC_one(self.y_train, result_dict['p_train'],
                        save_label='%s_TRAIN' % result_dict['model_label'],
                        result_path=RESULT_PATH)

        # 用这个来监测是否用test样本而不是用self.with_test是为了更好的延展性。可以plot任何模型结果的图
        p_test = result_dict.get('p_test', '')
        if with_test == False:
            y_all = self.y_train
        else:
            pl.ks_new(np.array(result_dict['p_test']), np.array(self.y_test), save_label='%s_TEST' % result_dict['model_label'], \
                      plot=True, result_path=RESULT_PATH)
            pl.print_AUC_one(self.y_test, result_dict['p_test'],
                             save_label='%s_TEST' % result_dict['model_label'],
                             result_path=RESULT_PATH)
            y_all = pd.concat([self.y_train, self.y_test])

        # cv
        pl.ks_new(np.array(result_dict['all_p_cv']), np.array(y_all), save_label='%s_%sFoldCV' % \
                  (result_dict['model_label'], result_dict['nfold']), \
                  plot=True, result_path=RESULT_PATH)
        pl.print_AUC_one(y_all, result_dict['all_p_cv'], save_label='%s_%sFoldCV' % \
                        (result_dict['model_label'], result_dict['nfold']),
                        result_path=RESULT_PATH)

    def plot_for_the_model_sw(self, sample_weight,y,RESULT_PATH, result_dict, with_test=True):
        """
        plot AUC， KS图 for train, test, cv

        Args:
        result_dict (dict): the dict returned by self.fit_model()。单独pass进来而不是直接用
            self.result 是为了此函数的延展性，可以plot统一个建模项目中任意一个版本的模型结果的图
            需包含key值为：'p_train', 'p_test', 'all_p_cv', 'model_label', 'nfold'
        sample_weight(series):样本权重
        """
        sample_weight=pd.DataFrame([sample_weight,y])
        sample_weight=sample_weight.T
        #sample_weight=pd.merge(sample_weight,y,left_index=True,right_index=True)
        pl.ks_new_sw(result_dict['p_train'], sample_weight, save_label='%s_TRAIN' % result_dict['model_label'], \
                  plot=True, result_path=RESULT_PATH)
        pl.print_AUC_one_sw(result_dict['p_train'],sample_weight,
                        save_label='%s_TRAIN' % result_dict['model_label'],
                        result_path=RESULT_PATH)
        # 用这个来监测是否用test样本而不是用self.with_test是为了更好的延展性。可以plot任何模型结果的图
        p_test = result_dict.get('p_test', '')
        if with_test == False:
            y_all = self.y_train
        else:
            pl.ks_new_sw(result_dict['p_test'], sample_weight, save_label='%s_TEST' % result_dict['model_label'], \
                      plot=True, result_path=RESULT_PATH)
            pl.print_AUC_one_sw(result_dict['p_test'],sample_weight,
                             save_label='%s_TEST' % result_dict['model_label'],
                             result_path=RESULT_PATH)
            y_all = pd.concat([self.y_train, self.y_test])

        # cv
        pl.ks_new_sw(result_dict['all_p_cv'], sample_weight, save_label='%s_%sFoldCV' % \
                  (result_dict['model_label'], result_dict['nfold']), \
                  plot=True, result_path=RESULT_PATH)
        pl.print_AUC_one_sw(result_dict['all_p_cv'],sample_weight, save_label='%s_%sFoldCV' % \
                        (result_dict['model_label'], result_dict['nfold']),
                        result_path=RESULT_PATH)


    def model_save(self, DATA_PATH, RESULT_PATH):
        """
        当完成self.fit_model() and/or self.plot_for_the_model(), 觉得模型可以当做备选的时候
        save model 结果，把系数单独存成excel，方便后面制作文档
        """
        self.train_stat.to_excel(os.path.join(RESULT_PATH, '%s_train_coef.xlsx' % self.model_label))
        if self.with_test:
            self.test_stat.to_excel(os.path.join(RESULT_PATH, '%s_test_coef.xlsx' % self.model_label))
        save_data_to_pickle(self.model_result, DATA_PATH, '%s模型结果.pkl' % self.model_label)


class XGBModel(object):
    def __init__(self, var_dict, DATA_PATH, RESULT_PATH, model_label_version, y_train, y_test=None,
                params_selection_method=['XGBExp'], nfold=5, x_train=None, x_test=None, use_origin=False):
        """
        params_selection_method (list): ['XGBExp', 'XGBGrid', 'XGBRandom']
        """
        self.var_dict = var_dict
        self.DATA_PATH = DATA_PATH
        self.RESULT_PATH = RESULT_PATH
        if not use_origin:
            xgboost_data_spec_dict = load_data_from_pickle(DATA_PATH, '%s_XGBoost输出数据和分箱明细.pkl' % model_label_version)
            self.X_train_xgboost = xgboost_data_spec_dict['X_train_xgboost']
            self.X_test_xgboost = xgboost_data_spec_dict.get('X_test_xgboost', '')
        else:
            if (x_train.shape[0] != y_train.shape[0]) or (x_test.shape[0] != y_test.shape[0]) or (np.prod(x_train.shape) == 0) or (np.prod(x_test.shape) == 0):
                print('x_train.shape == {}, y_train.shape == {}, x_test.shape == {}, y_test.shape == {}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
            self.X_train_xgboost = x_train
            self.X_test_xgboost = x_test
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
            self.fit_model(self.param_experience,
                           '%sXGBExp' % self.model_label_version,
                           self.NFOLD)

        if 'XGBGrid' in self.params_selection_method:
            logging.info("XGBoost模型使用GridSearch调参所选最优参数建模")
            self.fit_model(self.gridsearch_params,
                           '%sXGBGrid' % self.model_label_version,
                           self.NFOLD)

        if 'XGBRandom' in self.params_selection_method:
            logging.info("XGBoost模型使用RandomSearch调参所选最优参数建模")
            self.fit_model(self.randomsearch_params,
                          '%sXGBRandom' % self.model_label_version,
                          self.NFOLD)

        if 'XGBHyperopt' in self.params_selection_method:
            logging.info("XGBoost模型使用Hyperopt调参所选最优参数建模")
            self.fit_model(self.hyperopt_params,
                          '%sXGBHyperopt' % self.model_label_version,
                          self.NFOLD)

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
                logging.info('开始Grid调参')
                self.gridsearch_params = fs.ParamsTuning()\
                        .xgboost_gridsearch(self.X_train_xgboost[self.selected], self.y_train,
                                            self.X_test_xgboost[self.selected], self.y_test,
                                            self.NFOLD, params_range)

                self.gridsearch_params = fs.ParamsTuning()\
                        .xgboost_tree(self.X_train_xgboost[self.selected], self.y_train,
                                      self.gridsearch_params)
                logging.info('Grid调参完成！')
                save_data_to_json(self.gridsearch_params, self.DATA_PATH, 'xgboost_params_gridsearch.json')
                logging.info("Grid最优参数存储于%s" % os.path.join(self.DATA_PATH, 'xgboost_params_gridsearch.json'))

            if 'XGBRandom' in self.params_selection_method:
                logging.info("开始Random调参")
                self.randomsearch_params = fs.ParamsTuning()\
                        .xgboost_randomgridsearch(self.X_train_xgboost[self.selected], self.y_train,
                                                  self.X_test_xgboost[self.selected], self.y_test,
                                                  self.NFOLD, params_range)

                self.randomsearch_params = fs.ParamsTuning()\
                        .xgboost_tree(self.X_train_xgboost[self.selected], self.y_train,
                                      self.randomsearch_params)
                logging.info('Random调参完成！')
                save_data_to_json(self.randomsearch_params, self.DATA_PATH, 'xgboost_params_randomsearch.json')
                logging.info("Random最优参数存储于%s" % os.path.join(self.DATA_PATH, 'xgboost_params_randomsearch.json'))

            if 'XGBHyperopt' in self.params_selection_method:
                logging.info("开始Hyperopt超参调参")
                self.hyperopt_params = fs.ParamsTuning()\
                        .xgboost_hyperopt(self.X_train_xgboost[self.selected], self.y_train,
                                                  self.X_test_xgboost[self.selected], self.y_test,
                                                  space, isPrint, fit_params)
                self.hyperopt_params = fs.ParamsTuning()\
                        .xgboost_tree(self.X_train_xgboost[self.selected], self.y_train,
                                      self.hyperopt_params)
                logging.info('Hyperopt调参完成！')
                save_data_to_json(str(self.hyperopt_params), self.DATA_PATH, 'xgboost_params_hyperopt.json')
                logging.info("Hyperopt最优参数存储于%s" % os.path.join(self.DATA_PATH, 'xgboost_params_hyperopt.json'))

        else:
            X_train, X_test, y_train, y_test = train_test_split(self.X_train_xgboost[self.selected], self.y_train, test_size=0.3, random_state=0)

            if 'XGBGrid' in self.params_selection_method:
                logging.info('开始Grid调参')
                self.gridsearch_params = fs.ParamsTuning()\
                        .xgboost_gridsearch(X_train, y_train, X_test, y_test,
                                            self.NFOLD, params_range)

                self.gridsearch_params = fs.ParamsTuning()\
                        .xgboost_tree(X_train, y_train, self.gridsearch_params)
                logging.info('Grid调参完成！')
                save_data_to_json(self.gridsearch_params, self.DATA_PATH, 'xgboost_params_gridsearch.json')
                logging.info("Grid调参最优参数存储于%s" % os.path.join(self.DATA_PATH, 'xgboost_params_gridsearch.json'))

            if 'XGBRandom' in self.params_selection_method:
                logging.info("开始Random调参")
                self.randomsearch_params = fs.ParamsTuning()\
                        .xgboost_randomgridsearch(X_train, y_train, X_test, y_test,
                                            self.NFOLD, params_range)

                self.randomsearch_params = fs.ParamsTuning()\
                        .xgboost_tree(X_train, y_train, self.randomsearch_params)
                logging.info('Random调参完成！')
                save_data_to_json(self.randomsearch_params, self.DATA_PATH, 'xgboost_params_randomsearch.json')
                logging.info("Random调参最优参数存储于%s" % os.path.join(self.DATA_PATH, 'xgboost_params_randomsearch.json'))

            if 'XGBHyperopt' in self.params_selection_method:
                logging.info("开始Hyperopt调参")
                self.hyperopt_params = fs.ParamsTuning()\
                        .xgboost_hyperopt(self.X_train_xgboost[self.selected], self.y_train,
                                                  self.X_test_xgboost[self.selected], self.y_test,
                                                  space, isPrint, fit_params)
                self.hyperopt_params = fs.ParamsTuning()\
                        .xgboost_tree(self.X_train_xgboost[self.selected], self.y_train,
                                      self.hyperopt_params)
                logging.info('Hyperopt调参完成！')
                save_data_to_json(self.hyperopt_params, self.DATA_PATH, 'xgboost_params_hyperopt.json')
                logging.info("Hyperopt调参最优参数存储于%s" % os.path.join(self.DATA_PATH, 'xgboost_params_hyperopt.json'))

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



    def fit_model(self, param, model_label, nfold=5):
        if len(self.X_test_xgboost) > 0:
            all_X = pd.concat([self.X_train_xgboost, self.X_test_xgboost])[self.selected]
            all_y = pd.concat([self.y_train, self.y_test])
        else:
            all_X = self.X_train_xgboost[self.selected].copy()
            all_y = self.y_train

        logging.info("XGB模型在参数已调整好后，使用全部TRAIN&TEST数据集建模")
        all_p, xgbmodel_result, importance_df = self.xgboost(all_X, all_y, param)

        if len(self.X_test_xgboost) > 0:
            p_train = all_p.loc[self.X_train_xgboost.index].copy()
            p_test = all_p.loc[self.X_test_xgboost.index].copy()
        else:
            p_train = all_p
            p_test = p_train

        final_cols = xgbmodel_result.__dict__['feature_names']
        all_p_cv = self.xgboost_cv(all_X[final_cols], all_y, param, nfold)

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

class XGBModel_sw(object):
    def __init__(self, var_dict, DATA_PATH, RESULT_PATH, model_label_version, y_train, y_test,sample_weight_train,sample_weight_test,\
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
        self.sample_weight_train=sample_weight_train
        self.sample_weight_test=sample_weight_test
        self.params_selection_method = params_selection_method
        self.NFOLD = nfold
        self.model_label_version = model_label_version

    def run(self, params_range=None, param_experience=None, xgb_select=True):
        if xgb_select:
            self.xgb_select_sw()
        else:
            self.xgb_select_sw(auto=False)
        self.get_params_sw(params_range, param_experience)

        if 'XGBExp' in self.params_selection_method:
            logging.info("XGBoost模型使用经验参数建模")
            self.fit_model_sw(self.param_experience,
                           '%sXGBExp' % self.model_label_version,
                           self.NFOLD)

        if 'XGBGrid' in self.params_selection_method:
            logging.info("XGBoost模型使用GridSearch调参所选最优参数建模")
            self.fit_model_sw(self.gridsearch_params,
                           '%sXGBGrid' % self.model_label_version,
                           self.NFOLD)

        if 'XGBRandom' in self.params_selection_method:
            logging.info("XGBoost模型使用RandomSearch调参所选最优参数建模")
            self.fit_model_sw(self.randomsearch_params,
                          '%sXGBRandom' % self.model_label_version,
                          self.NFOLD)


    def xgb_select_sw(self,auto=True):
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
        train_xgb_rank = fs.FeatureSelection().xgboost_sw(self.X_train_xgboost, self.y_train,self.sample_weight_train)
        selected = _obtain_select(train_xgb_rank)

        if len(self.X_test_xgboost) > 10000:
            test_xgb_rank = fs.FeatureSelection().xgboost_sw(self.X_test_xgboost, self.y_test,self.sample_weight_test)
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


    def get_params_sw(self, params_range=None, param_experience=None):
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
                    .xgboost_tree_sw(self.X_train_xgboost[self.selected], self.y_train,self.sample_weight_train,
                                  self.param_experience)
            save_data_to_json(self.param_experience, self.DATA_PATH, 'xgboost_params_experience.json')
            logging.info("经验参数存储于%s" % os.path.join(self.DATA_PATH, 'xgboost_params_experience.json'))

        if len(self.X_test_xgboost) > 0:
            if 'XGBGrid' in self.params_selection_method:
                logging.info('开始网格调参')
                self.gridsearch_params = fs.ParamsTuning()\
                        .xgboost_gridsearch_sw(self.X_train_xgboost[self.selected], self.y_train,
                                            self.X_test_xgboost[self.selected], self.y_test,self.sample_weight_train,self.sample_weight_test,
                                            self.NFOLD, params_range)

                self.gridsearch_params = fs.ParamsTuning()\
                        .xgboost_tree_sw(self.X_train_xgboost[self.selected], self.y_train,self.sample_weight_train,
                                      self.gridsearch_params)
                logging.info('网格调参完成！')
                save_data_to_json(self.gridsearch_params, self.DATA_PATH, 'xgboost_params_gridsearch.json')
                logging.info("网格调参最优参数存储于%s" % os.path.join(self.DATA_PATH, 'xgboost_params_gridsearch.json'))

            if 'XGBRandom' in self.params_selection_method:
                logging.info("开始随机调参")
                self.randomsearch_params = fs.ParamsTuning()\
                        .xgboost_randomgridsearch_sw(self.X_train_xgboost[self.selected], self.y_train,
                                                  self.X_test_xgboost[self.selected], self.y_test,self.sample_weight_train,self.sample_weight_test,
                                                  self.NFOLD, params_range)

                self.randomsearch_params = fs.ParamsTuning()\
                        .xgboost_tree_sw(self.X_train_xgboost[self.selected], self.y_train,self.sample_weight_train,
                                      self.randomsearch_params)
                logging.info('随机调参完成！')
                save_data_to_json(self.randomsearch_params, self.DATA_PATH, 'xgboost_params_randomsearch.json')
                logging.info("随机调参最优参数存储于%s" % os.path.join(self.DATA_PATH, 'xgboost_params_randomsearch.json'))
        else:
            #下一步优化
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





    def xgboost_sw(self, X, y,sample_weight, param):
        """
        training and prediction
        params 应该已经调参完成了
        """
        dtrain = xgb.DMatrix(X, label=y,weight=sample_weight)
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
            return self.xgboost_sw(X.drop(na_cols, 1), y,sample_weight, param)
        else:
            return p_train, xgbmodel, importance_df



    def xgboost_cv_sw(self, X, y,sample_weight,param, folds=5):
        kf = KFold(n_splits=folds, shuffle=True)
        p_test_list = []
        for train_index , test_index in kf.split(X):
            _, xgbmodel, _ = self.xgboost_sw(X.iloc[train_index], y.iloc[train_index],sample_weight.iloc[train_index],
                                            param)
            dtest = xgb.DMatrix(X.iloc[test_index], label=y.iloc[test_index],weight=sample_weight.iloc[train_index])
            p_test = xgbmodel.predict(dtest)
            p_test = pd.Series(p_test, index=y.iloc[test_index].index)
            p_test_list.append(p_test)
        all_p_test = pd.concat(p_test_list)
        return all_p_test.loc[y.index]



    def fit_model_sw(self, param, model_label, nfold=5):
        if len(self.X_test_xgboost) > 0:
            all_X = pd.concat([self.X_train_xgboost, self.X_test_xgboost])[self.selected]
            all_y = pd.concat([self.y_train, self.y_test])
            all_sample_weight=pd.concat([self.sample_weight_train, self.sample_weight_test])
        else:
            all_X = self.X_train_xgboost[self.selected].copy()
            all_y = self.y_train
            all_sample_weight=self.sample_weight_train

        logging.info("XGB模型在参数已调整好后，使用全部TRAIN&TEST数据集建模")
        all_p, xgbmodel_result, importance_df = self.xgboost_sw(all_X, all_y,all_sample_weight,param)

        if len(self.X_test_xgboost) > 0:
            p_train = all_p.loc[self.X_train_xgboost.index].copy()
            p_test = all_p.loc[self.X_test_xgboost.index].copy()
        else:
            p_train = all_p
            p_test = p_train

        final_cols = xgbmodel_result.__dict__['feature_names']
        all_p_cv = self.xgboost_cv_sw(all_X[final_cols], all_y,all_sample_weight, param, nfold)

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
