# encoding=utf-8
import os
import json
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from functools import reduce
try:
    import xgboost as xgb
except:
    pass
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LassoCV, Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2

import utils3.metrics as mt
import utils3.misc_utils as mu
import utils3.plotting as pl
# 如果不安装超参调节包，hyperopt则无法运行
try:
    from hyperopt import fmin, tpe, hp, partial
    from hyperopt.pyll.stochastic.sample import sample as hp_sample
except:
    pass
from sklearn.metrics import mean_squared_error, zero_one_loss
from sklearn import metrics

class ParamsTuning(object):
    def __init__(self):
        self.xgboost_params_default = {
            'objective': 'binary:logistic',
            'silent': 0,
            'nthread': 4,
            'eval_metric': 'auc',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }


    def xgboost_tree(self, X, y, param=None):
        logging.info('xgb_cv start')
        dtrain = xgb.DMatrix(X, label=y)
        if param == None:
            param = {
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
        cv_res = xgb.cv(
            param,
            dtrain,
            num_boost_round=1000,
            nfold=5,
            verbose_eval=10,
            show_stdv=False,
            seed=1,
            early_stopping_rounds=30
        )
        param['num_boost_round'] = cv_res.shape[0]
        return param

    def xgboost_tree_sw(self, X, y,sample_weight, param=None):
        if sample_weight is None:
            dtrain = xgb.DMatrix(X, label=y)
        else:
            #print(type(sample_weight))
            sample_weight_sw=np.array(sample_weight)
            dtrain=xgb.DMatrix(X,label=y,weight=sample_weight_sw)
        if param == None:
            param = {
                'max_depth': 2,
                'min_samples_leaf ': int(len(X)*0.02),
                'eta': 0.1,
                'objective': 'binary:logistic',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'silent': 0,
                'eval_metric':'auc'
            }
        cv_res = xgb.cv(
            param,
            dtrain,
            num_boost_round=1000,
            nfold=5,
            verbose_eval=10,
            show_stdv=False,
            seed=1,
            early_stopping_rounds=30,
        )
        # best n tree
        param['num_boost_round'] = cv_res.shape[0]
        return param

    def xgboost_gridsearch(self, X_train, y_train, X_test, y_test, NFOLD=5, param=None):
        """
        网格搜索xgb的最佳参数；如果有test数据，则在gridsearch每一组参数训练时，
        XGBClassifier fit 时使用early_stopping，以在test上的效果不再提速时结束训练
        """
        if param == None:
            param = {
                'learning_rate': [0.05,0.1,0.2],
                'max_depth': [2, 3],
                'gamma': [0, 0.1],
                'min_child_weight':[1,3,10,20],
                'subsample': np.linspace(0.4, 0.7, 3),
                'colsample_bytree': np.linspace(0.4, 0.7, 3)
            }

        # fit_parameters = {
        #     'early_stopping_rounds': 30,
        #     'eval_metric': ['auc'],
        #     'eval_set': [[X_test, y_test]],
        #     'verbose': [False]
        # }

        # logging.info('fit_parameters==== {}'.format(fit_parameters))

        clf_obj = xgb.XGBClassifier(objective= 'binary:logistic', random_state=1,\
                                    n_estimators=1000, subsample=0.8,
                                    colsample_bytree=0.8)
        cv_obj = GridSearchCV(clf_obj, param, scoring='roc_auc', cv=NFOLD, verbose=8)
        cv_obj.fit(X_train, y_train)
        optimal_params = cv_obj.cv_results_['params'][cv_obj.cv_results_['mean_test_score'].argmax()]

        logging.info("GridSearch搜索的组合:{}".format(len(cv_obj.cv_results_["params"])))
        logging.info("GridSearch最佳mean_test_score:{}".format(cv_obj.cv_results_["mean_test_score"].max()))
        logging.info("GridSearch最佳参数:{}".format(optimal_params))

        optimal_params = dict(optimal_params, **self.xgboost_params_default)
        if 'learning_rate' in optimal_params:
            optimal_params['eta'] = optimal_params.pop('learning_rate')

        return optimal_params

    def xgboost_gridsearch_sw(self, X_train, y_train, X_test, y_test, sample_weight_train,sample_weight_test,NFOLD=5, param=None):
        """
        网格搜索xgb的最佳参数；如果有test数据，则在gridsearch每一组参数训练时，
        XGBClassifier fit 时使用early_stopping，以在test上的效果不再提速时结束训练
        """
        if param == None:
            param = {
                'learning_rate': [0.05,0.1,0.2],
                'max_depth': [2, 3],
                'gamma': [0, 0.1],
                'min_child_weight':[1,3,10,20],
                # 'subsample': np.linspace(0.4, 0.7, 3),
                # 'colsample_bytree': np.linspace(0.4, 0.7, 3),
            }

        fit_params = {
            "early_stopping_rounds": 30,
            "eval_metric": "auc",
            "eval_set": [(X_test, y_test)],
            "verbose": False,
            "sample_weight_eval_set":[sample_weight_test]
        }

        clf_obj = xgb.XGBClassifier(objective= 'binary:logistic', random_state=1,\
                                    n_estimators=1000, subsample=0.8,
                                    colsample_bytree=0.8)
        cv_obj = GridSearchCV(clf_obj, param, scoring='roc_auc', cv=NFOLD,
                              fit_params=fit_params)
        cv_obj.fit(X_train, y_train,sample_weight=sample_weight_train)
        optimal_params = cv_obj.cv_results_['params'][cv_obj.cv_results_['mean_test_score'].argmax()]

        logging.info("GridSearch搜索的组合:{}".format(len(cv_obj.cv_results_["params"])))
        logging.info("GridSearch最佳mean_test_score:{}".format(cv_obj.cv_results_["mean_test_score"].max()))
        logging.info("GridSearch最佳参数:{}".format(optimal_params))

        optimal_params = dict(optimal_params, **self.xgboost_params_default)
        if 'learning_rate' in optimal_params:
            optimal_params['eta'] = optimal_params.pop('learning_rate')

        return optimal_params


    def xgboost_randomgridsearch(self, X_train, y_train, X_test, y_test,
                            NFOLD=5, param=None):
        """
        随机搜索xgb的最佳参数；如果有test数据，则在gridsearch每一组参数训练时，
        XGBClassifier fit 时使用early_stopping，以在test上的效果不再提速时结束训练
        """
        if param == None:
            param = {
                'learning_rate': [0.05,0.1,0.2],
                'max_depth': [2, 3],
                'gamma': [0, 0.1],
                'min_child_weight':[1,3,10,20],
                # 'subsample': np.linspace(0.4, 0.7, 3),
                # 'colsample_bytree': np.linspace(0.4, 0.7, 3),
            }

#        fit_params = {
#            "early_stopping_rounds": 30,
#            "eval_metric": "auc",
#            "eval_set": [(X_test, y_test)],
#            "verbose": False,
#        }

        clf_obj = xgb.XGBClassifier(objective= 'binary:logistic', random_state=1,\
                                    n_estimators=1000, subsample=0.8,
                                    colsample_bytree=0.8)
        cv_obj = RandomizedSearchCV(clf_obj, param, scoring='roc_auc', cv=NFOLD)
        cv_obj.fit(X_train, y_train)
        optimal_params = cv_obj.cv_results_['params'][cv_obj.cv_results_['mean_test_score'].argmax()]

        logging.info("RandomSearch搜索的组合:{}".format(len(cv_obj.cv_results_["params"])))
        logging.info("RandomSearch最佳mean_test_score:{}".format(cv_obj.cv_results_["mean_test_score"].max()))
        logging.info("RandomSearch最佳参数:{}".format(optimal_params))

        optimal_params = dict(optimal_params, **self.xgboost_params_default)
        if 'learning_rate' in optimal_params:
            optimal_params['eta'] = optimal_params.pop('learning_rate')

        return optimal_params

    def xgboost_randomgridsearch_sw(self, X_train, y_train, X_test, y_test,sample_weight_train,sample_weight_test,
                            NFOLD=5, param=None):
        """
        随机搜索xgb的最佳参数；如果有test数据，则在gridsearch每一组参数训练时，
        XGBClassifier fit 时使用early_stopping，以在test上的效果不再提速时结束训练
        """
        if param == None:
            param = {
                'learning_rate': [0.05,0.1,0.2],
                'max_depth': [2, 3],
                'gamma': [0, 0.1],
                'min_child_weight':[1,3,10,20],
            }

        fit_params = {
            "early_stopping_rounds": 30,
            "eval_metric": "auc",
            "eval_set": [(X_test, y_test)],
            "verbose": False,
            "sample_weight_eval_set":[sample_weight_test]
        }

        clf_obj = xgb.XGBClassifier(objective= 'binary:logistic', random_state=1,\
                                    n_estimators=1000, subsample=0.8,
                                    colsample_bytree=0.8)
        cv_obj = RandomizedSearchCV(clf_obj, param, scoring='roc_auc', cv=NFOLD,
                              fit_params=fit_params)
        cv_obj.fit(X_train, y_train,sample_weight=sample_weight_train)
        optimal_params = cv_obj.cv_results_['params'][cv_obj.cv_results_['mean_test_score'].argmax()]

        logging.info("RandomSearch搜索的组合:{}".format(len(cv_obj.cv_results_["params"])))
        logging.info("RandomSearch最佳mean_test_score:{}".format(cv_obj.cv_results_["mean_test_score"].max()))
        logging.info("RandomSearch最佳参数:{}".format(optimal_params))

        optimal_params = dict(optimal_params, **self.xgboost_params_default)
        if 'learning_rate' in optimal_params:
            optimal_params['eta'] = optimal_params.pop('learning_rate')

        return optimal_params

    def xgboost_hyperopt(self, X_train, y_train, X_test, y_test, space=None, isPrint=True, fit_params=None):
        """
        Use hyperopt to get best params

        Args:
        X_train (pd.DataFrame()): X_train data frame that contains the variables
        y_train (pd.Series()): y_train data, labeling the performance
        X_test (pd.DataFrame()): X_test data frame that contains the variables
        y_test (pd.Series()): y_test data, labeling the performance

        space (dict): hyperopt dict
        isPrint (bool): whether print the params
        fit_params(dict): fit_params dict

        Returns:
        best(dict): best hyperopt params
        """
        # 再分出一个x_predict集合防止过拟合
        x_train_2, x_predict, y_train_2, y_predict = train_test_split(X_train, y_train, test_size=0.1, random_state=100)
        dtrain = xgb.DMatrix(data=x_train_2,label=y_train_2)
        dtest = xgb.DMatrix(data=X_test,label=y_test)
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        
        if fit_params == None:
            # 信贷领域最基本的评估以及拟合形式
            fit_params = {'eval_metric' :'auc','objective': 'binary:logistic'}
        
        if space == None:
            space = {'max_depth': hp.randint('max_depth', 10),
                     'n_estimators': hp.randint('n_estimators', 300),
                     'scale_pos_weight': hp.randint('scale_pos_weight', 2),
                     'subsample': hp.randint('subsample', 4),
                     'min_child_weight': hp.randint('min_child_weight', 4),
                     'learning_rate': hp.uniform('learning_rate', 1e-1, 2e-1),
                     'gamma': hp.uniform('gamma', 0, 2e-1),
                     'colsample_bytree': hp.uniform('colsample_bytree', low=1e-3, high=0.8)
                     }
        
        # 内部函数做一些参数转换使用，防止从0开始的现象浪费时间
        def argsDict_tranform(argsDict, isPrint):
            argsDict['n_estimators'] = argsDict['n_estimators'] + 5
            argsDict['max_depth'] = argsDict['max_depth'] + 1
            argsDict['scale_pos_weight'] = argsDict['scale_pos_weight'] + 1
            if 'subsample' in argsDict.keys():
                argsDict['subsample'] = argsDict['subsample'] * 0.1 + 0.5
            if 'min_child_weight' in argsDict.keys():
                argsDict['min_child_weight'] = argsDict['min_child_weight'] + 1
#            argsDict['learning_rate'] = argsDict['learning_rate'] * 0.2 + 0.05
#            argsDict['colsample_bytree'] = argsDict['colsample_bytree'] * 0.05 + 0.3
            if isPrint:
                print(argsDict)
                    
            return argsDict
        
        # fmin函数的核心部分，最优化x_predict这个集合的auc
        def get_tranformer_score(tranformer):
            xrf = tranformer
            dpredict = xgb.DMatrix(x_predict)
            prediction = xrf.predict(dpredict, ntree_limit=xrf.best_ntree_limit)
            test_auc = metrics.roc_auc_score(y_predict,prediction)
            return -test_auc
        
        # 工厂函数，完成训练过程
        def xgboost_factory(argsDict):
            argsDict = argsDict_tranform(argsDict,isPrint)
            params = {i:j for i,j in argsDict.items()}
            for i,j in fit_params.items():
                params[i] = j
            xrf = xgb.train(params, dtrain, 300, evallist, early_stopping_rounds=10)

            return get_tranformer_score(xrf)
        
        # 开始使用hyperopt进行自动调参
        algo = partial(tpe.suggest, n_startup_jobs=1)
        best = fmin(xgboost_factory, space, algo=algo, max_evals=20)
        result_param = argsDict_tranform(best,isPrint)
        result_param.update(fit_params)
        return result_param

class FeatureSelection(object):
    def __init__(self):
        pass


    def random_forest(self, X, y, grid_search=False, param=None):
        """
        Use random forest importance to select feature

        Args:
        X (pd.DataFrame()): X data frame that contains the variables
        y (pd.Series()): y data, labeling the performance
        grid_search (bool): default=False. If set True, parameter tuning will be
            executed first by grid search and then calculate the importance rank
        param (dict): the dictionary specifying parameters to tune and ranges
            for grid search.

        Returns:
        imp_df (pd.DataFrame()): contains the variable names and corresponding
            importance metrics value

        Examples:
        result = FeatureSelection().random_forest(mytrain_x,mytrain_y)
        """
        optimal_params = {'n_estimators': 30, 'max_depth':2, 'min_samples_leaf': int(len(X)*0.05)}
        if grid_search:
            logging.info("entered random_forest cv")
            if param == None:
                param = {'n_estimators': [10, 20, 30, 40, 50],\
                          "max_depth": [2, 3, 4, 5],\
                          "min_samples_leaf": [int(len(X)*0.03), int(len(X)*0.05), int(len(X)*0.07)]
                          }

            clf_obj = RandomForestClassifier(criterion='entropy', random_state=1)
            cv_obj = GridSearchCV(clf_obj, param, scoring='roc_auc', cv=5)
            cv_obj.fit(X, y)
            optimal_params = cv_obj.cv_results_['params'][cv_obj.cv_results_['mean_test_score'].argmax()]


        rf = RandomForestClassifier(criterion="entropy", \
                                n_estimators=optimal_params['n_estimators'],\
                                max_depth=optimal_params['max_depth'],\
                                min_samples_leaf=optimal_params['min_samples_leaf'])
        rf.fit(X,y)
        importances = rf.feature_importances_
        imp_df = pd.DataFrame([X.columns,importances]).T
        imp_df.columns = ['var_code', 'random_forest_score']
        imp_df.loc[:, 'random_forest_rank'] = imp_df.random_forest_score.rank(ascending=False)
        return imp_df

    def random_forest_sw(self, X, y,sample_weight, grid_search=False, param=None):
        """
        Use random forest importance to select feature

        Args:
        X (pd.DataFrame()): X data frame that contains the variables
        y (pd.Series()): y data, labeling the performance
        sample_weight(pd.Series()):sample_weight
        grid_search (bool): default=False. If set True, parameter tuning will be
            executed first by grid search and then calculate the importance rank
        param (dict): the dictionary specifying parameters to tune and ranges
            for grid search.

        Returns:
        imp_df (pd.DataFrame()): contains the variable names and corresponding
            importance metrics value

        Examples:
        result = FeatureSelection().random_forest(mytrain_x,mytrain_y)
        """
        optimal_params = {'n_estimators': 30, 'max_depth':2, 'min_samples_leaf': int(len(X)*0.05)}
        if grid_search:
            logging.info("entered random_forest cv")
            if param == None:
                param = {'n_estimators': [10, 20, 30, 40, 50],\
                          "max_depth": [2, 3, 4, 5],\
                          "min_samples_leaf": [int(len(X)*0.03), int(len(X)*0.05), int(len(X)*0.07)]
                          }

            clf_obj = RandomForestClassifier(criterion='entropy', random_state=1)
            cv_obj = GridSearchCV(clf_obj, param, scoring='roc_auc', cv=5)
            if sample_weight is None:
                cv_obj.fit(X, y)
            else:
                sample_weight_sw=np.array(sample_weight)
                cv_obj.fit(X,y,sample_weight=sample_weight_sw)
            optimal_params = cv_obj.cv_results_['params'][cv_obj.cv_results_['mean_test_score'].argmax()]


        rf = RandomForestClassifier(criterion="entropy", \
                                n_estimators=optimal_params['n_estimators'],\
                                max_depth=optimal_params['max_depth'],\
                                min_samples_leaf=optimal_params['min_samples_leaf'])
        if sample_weight is None:
            rf.fit(X,y)
        else:
            sample_weight_sw=np.array(sample_weight)
            rf.fit(X,y,sample_weight=sample_weight_sw)
        importances = rf.feature_importances_
        imp_df = pd.DataFrame([X.columns,importances]).T
        imp_df.columns = ['var_code', 'random_forest_score']
        imp_df.loc[:, 'random_forest_rank'] = imp_df.random_forest_score.rank(ascending=False)
        return imp_df


    # NOTE：这个速度太慢了，所以先不弄CV调参了
    def svm(self, X, y):
        """
        Use svm importance to select feature

        Args:
        X (pd.DataFrame()): X data frame that contains the variables
        y (pd.Series()): y data, labeling the performance

        Returns:
        imp_df (pd.DataFrame()): contains the variable names and corresponding
            importance metrics value

        Examples:
        result = FeatureSelection().svm(mytrain_x,mytrain_y)
        """
        clf = SVC(C=1.0, class_weight=None, kernel='linear')
        clf.fit(X,y)
        [importances] = clf.coef_
        imp_df = pd.DataFrame([X.columns,importances]).T
        imp_df.columns = ['var_code', 'svm_score']
        imp_df.loc[:, 'svm_rank'] = imp_df.svm_score.rank(ascending=False)
        return imp_df

    def svm_sw(self, X, y,sample_weight):
        """
        Use svm importance to select feature

        Args:
        X (pd.DataFrame()): X data frame that contains the variables
        y (pd.Series()): y data, labeling the performance

        Returns:
        imp_df (pd.DataFrame()): contains the variable names and corresponding
            importance metrics value

        Examples:
        result = FeatureSelection().svm(mytrain_x,mytrain_y)
        """
        clf = SVC(C=1.0, class_weight=None, kernel='linear')
        if sample_weight is None:
            clf.fit(X,y)
        else:
            sample_weight_sw=np.array(sample_weight)
            clf.fit(X,y,sample_weight=sample_weight_sw)
        [importances] = clf.coef_
        imp_df = pd.DataFrame([X.columns,importances]).T
        imp_df.columns = ['var_code', 'svm_score']
        imp_df.loc[:, 'svm_rank'] = imp_df.svm_score.rank(ascending=False)
        return imp_df


    def xgboost(self, X, y, grid_search=False, param=None):
        """
        Use xgboost importance to select feature

        Args:
        X (pd.DataFrame()): X data frame that contains the variables.
            variable names should not contain [, ] or <, replace with (, ) and lt
        y (pd.Series()): y data, labeling the performance
        grid_search (bool): default=False. If set True, parameter tuning will be
            executed first by grid search and then calculate the importance rank
        param (dict): the dictionary specifying parameters to tune and ranges
            for grid search.

        Returns:
        imp_df (pd.DataFrame()): contains the variable names and the corresponding
            importance metric's values

        Examples:
        result = FeatureSelection().xgboost(mytrain_x,mytrain_y)
        """
        optimal_params = {
            # 不用调整的parameter
            'objective': 'binary:logistic', 
            'silent': 0, 
            'nthread': 4,
            'eval_metric': 'auc',
            # 可调整的parameters
            'eta': 0.1,
            'max_depth': 3,
            'gamma': 0, #minimum loss reduction required to make a further partition on a leaf node of the tree
            # 'min_child_weight': 20,
            'min_samples_leaf ': int(len(X)*0.02),
            #'subsample': 0.7,
            #'colsample_bytree': 0.7
        }
        if grid_search:
            logging.info("entered xgboost cv for gridsearch")
            optimal_params = ParamsTuning().xgboost_gridsearch(X, y, param=param)

        optimal_params = ParamsTuning().xgboost_tree(X, y, optimal_params)

        dtrain = xgb.DMatrix(X, label=y)
        bst = xgb.train(optimal_params, dtrain,
                        num_boost_round=optimal_params['num_boost_round'])
        # get_fscore() default importance_type='weight'
        importances_dict = bst.get_fscore()
        imp_df = pd.Series(importances_dict)\
                       .to_frame('xgboost_score')\
                       .reset_index()\
                       .rename(columns={'index': 'var_code'})
        imp_df.loc[:, 'xgboost_rank'] = imp_df.xgboost_score.rank(ascending=False)
        imp_df = imp_df.sort_values('xgboost_rank', ascending=True)
        return imp_df

    def xgboost_sw(self, X, y,sample_weight,grid_search=False, param=None):
        """
        Use xgboost importance to select feature

        Args:
        X (pd.DataFrame()): X data frame that contains the variables.
            variable names should not contain [, ] or <, replace with (, ) and lt
        y (pd.Series()): y data, labeling the performance
        grid_search (bool): default=False. If set True, parameter tuning will be
            executed first by grid search and then calculate the importance rank
        param (dict): the dictionary specifying parameters to tune and ranges
            for grid search.

        Returns:
        imp_df (pd.DataFrame()): contains the variable names and the corresponding
            importance metric's values

        Examples:
        result = FeatureSelection().xgboost(mytrain_x,mytrain_y)
        """
        optimal_params = {
            # 不用调整的parameter
            'objective': 'binary:logistic', 'silent': 0, 'nthread': 4,
            'eval_metric': 'auc',
            # 可调整的parameters
            'eta': 0.1,
            'max_depth': 3,
            'gamma': 0, # minimum loss reduction required to make a further partition on a leaf node of the tree
            # 'min_child_weight': 20,
            'min_samples_leaf ': int(len(X)*0.02),
            #'subsample': 0.7,
            #'colsample_bytree': 0.7
        }
        if grid_search:
            logging.info("entered xgboost cv for gridsearch")
            optimal_params = ParamsTuning().xgboost_gridsearch(X, y, param=param)

        optimal_params = ParamsTuning().xgboost_tree_sw(X, y,sample_weight, optimal_params)
        if sample_weight is None:
            dtrain = xgb.DMatrix(X, label=y)
        else:
            sample_weight_sw=np.array(sample_weight)
            dtrain = xgb.DMatrix(X, label=y,weight=sample_weight_sw)
        bst = xgb.train(optimal_params, dtrain,
                        num_boost_round=optimal_params['num_boost_round'])
        # get_fscore() default importance_type='weight'
        importances_dict = bst.get_fscore()
        imp_df = pd.Series(importances_dict)\
                       .to_frame('xgboost_score')\
                       .reset_index()\
                       .rename(columns={'index': 'var_code'})
        imp_df.loc[:, 'xgboost_rank'] = imp_df.xgboost_score.rank(ascending=False)
        imp_df = imp_df.sort_values('xgboost_rank', ascending=True)
        return imp_df


    def lasso(self, X, y, alpha=None):
        """
        Lasso variable selection

        Args:
        X (pd.DataFrame()): X data frame that contains the variables.
        y (pd.Series()): y data, labeling the performance
        alpha: default=None. If None, then will run CV to find the optimal alpha

        Returns:
        imp_df (pd.DataFrame()): contains the variable names and the corresponding
            importance metric's values

        Examples:
        result = FeatureSelection().lasso(mytrain_x,mytrain_y)
        """
        if pd.isnull(alpha):
            cv_lasso_obj = LassoCV(cv=10, normalize=True)
            cv_lasso_obj.fit(X, y)
            alpha = cv_lasso_obj.alpha_

        lasso_obj = Lasso(alpha=alpha)
        lasso_obj.fit(X, y)
        imp_df = pd.DataFrame([X.columns, lasso_obj.coef_]).T
        imp_df.columns = ['var_code', 'lasso_coef']
        imp_df.loc[:, 'lasso_rank'] = abs(imp_df.lasso_coef).rank(ascending=False)
        return imp_df



    def stepwise(self, X, y, logging_file, start_from=[], direction='FORWARD/BACKWARD',
                    lrt=True, lrt_threshold=0.05):
        """
        Use results of logistic regression to select features. It will go through
        stepwise based on the direction specified first and then check the
        significance of the variables using pvalue. Then check if there are variables
        with negative coefficients and remove it.

        Args:
        X (pd.DataFrame()): X data frame that contains the variables.
            variable names should not contain . or -, replace with _ and _
        y (pd.Series()): y data, labeling the performance
        logging_file (str): logging file object
        start_from (list): list of variable names that the model starts from,
            if not the null model
        direction (str): default='FOWARD/BACKWARD', forward first and then backward.
            ['FORWARD/BACKWARD', 'BACKWARD/FORWARD']
        lrt (bool): default = True. If set true, the model will consider likelihood
            ratio test result
        lrt_threshold (float): default = 0.05. The alpha value for likelihood ratio
            test.

        Returns:
        imp_df (pd.DataFrame()): contains the variable names and FORWARD AIC
            and ranking

        Examples:
        RESULT_PATH = '/Users/xiangyu/Documents/Seafile/Files/现金贷/Modeling/DRM-A v4/result/'
        logging.basicConfig(filename=os.path.join(RESULT_PATH, 'stepwise.log'), level=logging.INFO, filemode='w')
        LOG = logging.getLogger(__name__)
        f = FeatureSelection()
        result = f.stepwise(mytrain_x, mytrain_y, logging_file=LOG)
        """
        def _likelihood_ratio_test(ll_r, ll_f, lrt_threshold):
            # H0: reduced model is true. If rejected, then it means that the
            # full model is good. Need to add the candidate
            test_statistics = (-2 * ll_r) - (-2 * ll_f)
            p_value = 1 - chi2(df=1).cdf(test_statistics)
            return p_value <= lrt_threshold




        def _forward(dataset, logging_file, current_score, best_new_score,
                    remaining, selected, result_dict, reduced_loglikelihood):
            # TODO: 加上 f test like in SAS
            logging_file.info("While loop beginning current_score: %s" % current_score)
            logging_file.info("While loop beginning best_new_score: %s" % best_new_score)
            current_score = best_new_score
            aics_with_candidates = {}
            p_values_ok_to_add = []
            # 选择最好的变量to add
            for candidate in remaining:
                formula = "{} ~ {}".format('y', ' + '.join(selected + [candidate]))
                mod1 = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()
                # 只有新加指标的coefficient>0或者加上之后其他指标coefficient也>0
                if sum(mod1.params.loc[mod1.params.index != 'Intercept'] < 0) == 0 :
                    aics_with_candidates[candidate] = mod1.aic
                    full_loglikelihood = mod1.llf
                    if lrt:
                        p_values_ok = _likelihood_ratio_test(reduced_loglikelihood, full_loglikelihood, lrt_threshold)
                        if p_values_ok:
                            p_values_ok_to_add.append(candidate)

            # 只有通过likelihood ratio test的变量才会进行AIC比较进行选择
            candidate_scores = pd.Series(aics_with_candidates)
            if lrt:
                candidate_scores = candidate_scores.loc[p_values_ok_to_add]

            # 有变量pvalues显著 reject the reduced model and need to add the variable
            if not candidate_scores.empty:
                best = candidate_scores[candidate_scores == candidate_scores.min()]
                # best_new_score 被替换成新的加上这个变量的模型的AIC
                best_new_score = best.iloc[0]
                best_candidate = best.index.values[0]
            else:
                return None

            # 当加上变量的模型的AIC比当前模型的小时，选择加上变量的模型
            if current_score > best_new_score:
                logging_file.info('Best Variable to Add: %s' % best_candidate)
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                improvement_gained = current_score - best_new_score
                result_dict[best_candidate] = {'AIC_delta': improvement_gained, 'step': 'FORWARD'}
                formula = "{} ~ {}".format('y', ' + '.join(selected))
                mod2 = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()
                # loglikelihood of the reduced model
                reduced_loglikelihood = mod2.llf
                logging_file.info('FOWARD Step: AIC=%s' % mod2.aic)
                logging_file.info(mod2.summary())
                return current_score, best_new_score, result_dict, selected, remaining, reduced_loglikelihood
            else:
                return None



        def _backward(dataset, logging_file, current_score, best_new_score,
                    remaining, selected, result_dict, reduced_loglikelihood):
            # TODO: 加上 f test like in SAS
            logging_file.info("While loop beginning current_score: %s" % current_score)
            logging_file.info("While loop beginning best_new_score: %s" % best_new_score)
            current_score = best_new_score
            aics_with_candidates = {}
            p_values_ok_to_delete = []
            # 选择最差的to delete
            for candidate in selected:
                put_in_model = [i for i in selected if i != candidate]
                formula = "{} ~ {}".format('y', ' + '.join(put_in_model))
                mod1 = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()
                # 只有减去指标后留下的变量中没有变量coefficient是负数
                if sum(mod1.params.loc[mod1.params.index != 'Intercept'] < 0) == 0 :
                    aics_with_candidates[candidate] = mod1.aic
                    reduced_reduced_loglikelihood = mod1.llf
                    if lrt:
                        p_values_rejected = _likelihood_ratio_test(reduced_reduced_loglikelihood, reduced_loglikelihood, lrt_threshold)
                        # if not rejected, H0: reduced model is true, need to remove the variable
                        if not p_values_rejected:
                            p_values_ok_to_delete.append(candidate)

            # 只有通过likelihood ratio test的变量才会进行AIC比较进行选择
            candidate_scores = pd.Series(aics_with_candidates)
            if lrt:
                candidate_scores = candidate_scores.loc[p_values_ok_to_delete]

            # 有变量pvalues不显著 没有reject the reduced model, then need to delete the variable
            if not candidate_scores.empty:
                best = candidate_scores[candidate_scores == candidate_scores.min()]
                # best_new_score 被替换成新的减去这个变量的模型的AIC
                best_new_score = best.iloc[0]
                best_candidate = best.index.values[0]
            else:
                return None


            # 当减去变量的模型的AIC比当前模型的小时，选择减去变量的模型
            if current_score > best_new_score:
                logging_file.info('Best Variable to Delete: %s' % best_candidate)
                remaining.append(best_candidate)
                selected.remove(best_candidate)
                improvement_gained = current_score - best_new_score
                result_dict[best_candidate] = {'AIC_delta': improvement_gained, 'step': 'BACKWARD'}
                formula = "{} ~ {}".format('y', ' + '.join(selected))
                mod2 = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()
                # loglikelihood of the reduced model
                reduced_loglikelihood = mod2.llf
                logging_file.info('BACKWARD Step: AIC=%s' % mod2.aic)
                logging_file.info(mod2.summary())
                return current_score, best_new_score, result_dict, selected, remaining, reduced_loglikelihood
            else:
                return None


        # 因为之后的建模function不接受变量名含"."或者 "-"的，所以更换名字
        new_indexs = []
        for col in X.columns:
            new_col = 'header_' + col.replace('.','_').replace('-','_')
            new_indexs.append(new_col)

        dataset = X.copy()
        dataset.columns = new_indexs
        dataset.loc[:, 'y'] = y
        mapping = pd.DataFrame({'new_var_code': new_indexs, 'var_code': list(X.columns)})

        new_start_from = mapping.loc[mapping.var_code.isin(start_from), 'new_var_code'].tolist()
        logging_file.info(new_start_from)
        logging_file.info(mapping)

        if direction == 'FORWARD/BACKWARD':
            remaining = set(new_indexs)
            selected = []
            result_dict = {}

            # the first model is the null model with only Intercept, or if start_from is provided, it the model with these variables
            if len(start_from) == 0:
                init_model = smf.glm(formula="y ~ 1", data=dataset, family=sm.families.Binomial()).fit()
            else:
                formula = "{} ~ {}".format('y', ' + '.join(new_start_from))
                init_model = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()
                remaining = remaining - set(new_start_from)
                selected = new_start_from

            current_score, best_new_score = init_model.aic, init_model.aic
            reduced_loglikelihood = init_model.llf
            remaining = list(remaining)

            while current_score >= best_new_score:
                if remaining:
                    fwd_result = _forward(dataset, logging_file, current_score,
                                          best_new_score, remaining, selected,
                                          result_dict, reduced_loglikelihood)

                    if fwd_result != None:
                        current_score = fwd_result[0]
                        best_new_score = fwd_result[1]
                        result_dict = fwd_result[2]
                        selected = fwd_result[3]
                        remaining = fwd_result[4]
                        reduced_loglikelihood = fwd_result[5]
                    else:
                        logging_file.info("FORWARD completed and no variable selected to add")
                else:
                    fwd_result = None
                    logging_file.info("FORWARD has no more remaining variables to add")

                if len(selected) == 1:
                    continue
                elif len(selected) > 1:
                    bkd_result = _backward(dataset, logging_file, current_score,
                                          best_new_score, remaining, selected,
                                          result_dict, reduced_loglikelihood)

                    if bkd_result != None:
                        current_score = bkd_result[0]
                        best_new_score = bkd_result[1]
                        result_dict = bkd_result[2]
                        selected = bkd_result[3]
                        remaining = bkd_result[4]
                        reduced_loglikelihood = bkd_result[5]
                    else:
                        logging_file.info("BACKWARD completed and no variable selected to delete")
                else:
                    bkd_result = None
                    logging_file.info("BACKWARD has no more selected variables to delete")

                if not fwd_result and not bkd_result:
                    break




        elif direction == 'BACKWARD/FORWARD':
            remaining = []
            selected = list(set(new_indexs))
            result_dict = {}
            # the first model is the full model with all the variables
            formula = "{} ~ {}".format('y', ' + '.join(selected))
            init_model = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()
            current_score, best_new_score = init_model.aic, init_model.aic
            reduced_loglikelihood = init_model.llf

            while current_score >= best_new_score:
                if len(selected) == 1:
                    continue
                elif len(selected) > 1:
                    bkd_result = _backward(dataset, logging_file, current_score,
                                          best_new_score, remaining, selected,
                                          result_dict, reduced_loglikelihood)

                    if bkd_result != None:
                        current_score = bkd_result[0]
                        best_new_score = bkd_result[1]
                        result_dict = bkd_result[2]
                        selected = bkd_result[3]
                        remaining = bkd_result[4]
                        reduced_loglikelihood = bkd_result[5]
                    else:
                        logging_file.info("BACKWARD completed and no variable selected to delete")
                else:
                    bkd_result = None
                    logging_file.info("BACKWARD has no more selected variables to delete")


                if remaining:
                    fwd_result = _forward(dataset, logging_file, current_score,
                                          best_new_score, remaining, selected,
                                          result_dict, reduced_loglikelihood)

                    if fwd_result != None:
                        current_score = fwd_result[0]
                        best_new_score = fwd_result[1]
                        result_dict = fwd_result[2]
                        selected = fwd_result[3]
                        remaining = fwd_result[4]
                        reduced_loglikelihood = fwd_result[5]
                    else:
                        logging_file.info("FORWARD completed and no variable selected to add")
                else:
                    fwd_result = None
                    logging_file.info("FORWARD has no more remaining variables to add")

                if not fwd_result and not bkd_result:
                    break


        # results
        result_aic = pd.DataFrame(result_dict).transpose()\
                   .reset_index()\
                   .rename(columns={'index': 'new_var_code'})

        if selected:
            formula = "{} ~ {}".format('y', ' + '.join(selected))
            model = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()
            logging_file.info("##################### STEPWISE FINAL MODEL #####################################")
            logging_file.info(model.summary())
            pvalue = model.pvalues
            logging_file.info("Remove insignificant variables and those with negative coefficients")
            # 即使不显著，也要保留住start_from的变量
            pvalue = pvalue.drop(['Intercept'] + new_start_from)

            while pvalue.max() > 0.05 or sum(model.params.loc[model.params.index != 'Intercept'] < 0) > 0:
                while pvalue.max() > 0.05:
                    pvalue = pvalue.drop(pvalue.idxmax())
                    if len(pvalue) == 0:
                        break
                    formula = '{}~{}'.format('y', "+".join(pvalue.index))
                    model = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()
                    pvalue = model.pvalues
                    pvalue = pvalue.drop('Intercept')
                    logging_file.info("Max p-values: %s, variable: %s" % (pvalue.max(), pvalue.idxmax()))

                while sum(model.params.loc[model.params.index != 'Intercept'] < 0) > 0:
                    coefs = model.params.loc[model.params.index != 'Intercept'].copy()
                    negative_coef_variables = list(coefs.loc[coefs < 0].index)
                    pvalue = model.pvalues
                    pvalue = pvalue.drop('Intercept')
                    # drop coef为负中pvalue最大的
                    negative_coef_pvalue = pvalue.loc[negative_coef_variables]
                    logging_file.info("Max p-values: %s, variable: %s" % (negative_coef_pvalue.max(), negative_coef_pvalue.idxmax()))
                    pvalue = pvalue.drop(negative_coef_pvalue.idxmax())
                    formula = '{}~{}'.format('y', "+".join(pvalue.index))
                    model = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()


            logging_file.info("SIGNIFICANT MODEL")
            logging_file.info(model.summary())

            final_selected = model.pvalues.drop('Intercept').index
        else:
            final_selected = selected

        result_aic = result_aic.loc[result_aic.new_var_code.isin(final_selected)]
        imp_df = pd.merge(mapping, result_aic, how='left', on='new_var_code')
        imp_df.loc[:, 'final_selected'] = np.where(imp_df.new_var_code.isin(final_selected), 1, 0)
        imp_df = imp_df.drop('new_var_code', axis=1)

        return imp_df

    def stepwise_sw(self, X, y,sample_weight,logging_file, start_from=[], direction='FORWARD/BACKWARD',
                    lrt=True, lrt_threshold=0.05):
        """
        Use results of logistic regression to select features. It will go through
        stepwise based on the direction specified first and then check the
        significance of the variables using pvalue. Then check if there are variables
        with negative coefficients and remove it.

        Args:
        X (pd.DataFrame()): X data frame that contains the variables.
            variable names should not contain . or -, replace with _ and _
        y (pd.Series()): y data, labeling the performance
        sample_weight:(series) sample_weight
        logging_file (str): logging file object
        start_from (list): list of variable names that the model starts from,
            if not the null model
        direction (str): default='FOWARD/BACKWARD', forward first and then backward.
            ['FORWARD/BACKWARD', 'BACKWARD/FORWARD']
        lrt (bool): default = True. If set true, the model will consider likelihood
            ratio test result
        lrt_threshold (float): default = 0.05. The alpha value for likelihood ratio
            test.

        Returns:
        imp_df (pd.DataFrame()): contains the variable names and FORWARD AIC
            and ranking

        Examples:
        RESULT_PATH = '/Users/xiangyu/Documents/Seafile/Files/现金贷/Modeling/DRM-A v4/result/'
        logging.basicConfig(filename=os.path.join(RESULT_PATH, 'stepwise.log'), level=logging.INFO, filemode='w')
        LOG = logging.getLogger(__name__)
        f = FeatureSelection()
        result = f.stepwise(mytrain_x, mytrain_y, logging_file=LOG)
        """
        def _likelihood_ratio_test(ll_r, ll_f, lrt_threshold):
            # H0: reduced model is true. If rejected, then it means that the
            # full model is good. Need to add the candidate
            test_statistics = (-2 * ll_r) - (-2 * ll_f)
            p_value = 1 - chi2(df=1).cdf(test_statistics)
            return p_value <= lrt_threshold




        def _forward(dataset, logging_file, current_score, best_new_score,
                    remaining, selected, result_dict, reduced_loglikelihood,sample_weight):
            # TODO: 加上 f test like in SAS
            logging_file.info("While loop beginning current_score: %s" % current_score)
            logging_file.info("While loop beginning best_new_score: %s" % best_new_score)
            current_score = best_new_score
            aics_with_candidates = {}
            p_values_ok_to_add = []
            # 选择最好的变量to add
            for candidate in remaining:
                formula = "{} ~ {}".format('y', ' + '.join(selected + [candidate]))
                mod1 = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial(),freq_weights=sample_weight).fit()
                # 只有新加指标的coefficient>0或者加上之后其他指标coefficient也>0
                if sum(mod1.params.loc[mod1.params.index != 'Intercept'] < 0) == 0 :
                    aics_with_candidates[candidate] = mod1.aic
                    full_loglikelihood = mod1.llf
                    if lrt:
                        p_values_ok = _likelihood_ratio_test(reduced_loglikelihood, full_loglikelihood, lrt_threshold)
                        if p_values_ok:
                            p_values_ok_to_add.append(candidate)

            # 只有通过likelihood ratio test的变量才会进行AIC比较进行选择
            candidate_scores = pd.Series(aics_with_candidates)
            if lrt:
                candidate_scores = candidate_scores.loc[p_values_ok_to_add]

            # 有变量pvalues显著 reject the reduced model and need to add the variable
            if not candidate_scores.empty:
                best = candidate_scores[candidate_scores == candidate_scores.min()]
                # best_new_score 被替换成新的加上这个变量的模型的AIC
                best_new_score = best.iloc[0]
                best_candidate = best.index.values[0]
            else:
                return None

            # 当加上变量的模型的AIC比当前模型的小时，选择加上变量的模型
            if current_score > best_new_score:
                logging_file.info('Best Variable to Add: %s' % best_candidate)
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                improvement_gained = current_score - best_new_score
                result_dict[best_candidate] = {'AIC_delta': improvement_gained, 'step': 'FORWARD'}
                formula = "{} ~ {}".format('y', ' + '.join(selected))
                mod2 = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial(),freq_weights=sample_weight).fit()
                # loglikelihood of the reduced model
                reduced_loglikelihood = mod2.llf
                logging_file.info('FOWARD Step: AIC=%s' % mod2.aic)
                logging_file.info(mod2.summary())
                return current_score, best_new_score, result_dict, selected, remaining, reduced_loglikelihood
            else:
                return None



        def _backward(dataset, logging_file, current_score, best_new_score,
                    remaining, selected, result_dict, reduced_loglikelihood,sample_weight):
            # TODO: 加上 f test like in SAS
            logging_file.info("While loop beginning current_score: %s" % current_score)
            logging_file.info("While loop beginning best_new_score: %s" % best_new_score)
            current_score = best_new_score
            aics_with_candidates = {}
            p_values_ok_to_delete = []
            # 选择最差的to delete
            for candidate in selected:
                put_in_model = [i for i in selected if i != candidate]
                formula = "{} ~ {}".format('y', ' + '.join(put_in_model))
                mod1 = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial(),freq_weights=sample_weight).fit()
                # 只有减去指标后留下的变量中没有变量coefficient是负数
                if sum(mod1.params.loc[mod1.params.index != 'Intercept'] < 0) == 0 :
                    aics_with_candidates[candidate] = mod1.aic
                    reduced_reduced_loglikelihood = mod1.llf
                    if lrt:
                        p_values_rejected = _likelihood_ratio_test(reduced_reduced_loglikelihood, reduced_loglikelihood, lrt_threshold)
                        # if not rejected, H0: reduced model is true, need to remove the variable
                        if not p_values_rejected:
                            p_values_ok_to_delete.append(candidate)

            # 只有通过likelihood ratio test的变量才会进行AIC比较进行选择
            candidate_scores = pd.Series(aics_with_candidates)
            if lrt:
                candidate_scores = candidate_scores.loc[p_values_ok_to_delete]

            # 有变量pvalues不显著 没有reject the reduced model, then need to delete the variable
            if not candidate_scores.empty:
                best = candidate_scores[candidate_scores == candidate_scores.min()]
                # best_new_score 被替换成新的减去这个变量的模型的AIC
                best_new_score = best.iloc[0]
                best_candidate = best.index.values[0]
            else:
                return None


            # 当减去变量的模型的AIC比当前模型的小时，选择减去变量的模型
            if current_score > best_new_score:
                logging_file.info('Best Variable to Delete: %s' % best_candidate)
                remaining.append(best_candidate)
                selected.remove(best_candidate)
                improvement_gained = current_score - best_new_score
                result_dict[best_candidate] = {'AIC_delta': improvement_gained, 'step': 'BACKWARD'}
                formula = "{} ~ {}".format('y', ' + '.join(selected))
                mod2 = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial(),freq_weights=sample_weight).fit()
                # loglikelihood of the reduced model
                reduced_loglikelihood = mod2.llf
                logging_file.info('BACKWARD Step: AIC=%s' % mod2.aic)
                logging_file.info(mod2.summary())
                return current_score, best_new_score, result_dict, selected, remaining, reduced_loglikelihood
            else:
                return None


        # 因为之后的建模function不接受变量名含"."或者 "-"的，所以更换名字
        new_indexs = []
        for col in X.columns:
            new_col = 'header_' + col.replace('.','_').replace('-','_')
            new_indexs.append(new_col)

        dataset = X.copy()
        dataset.columns = new_indexs
        dataset.loc[:, 'y'] = y
        mapping = pd.DataFrame({'new_var_code': new_indexs, 'var_code': list(X.columns)})

        new_start_from = mapping.loc[mapping.var_code.isin(start_from), 'new_var_code'].tolist()
        logging_file.info(new_start_from)
        logging_file.info(mapping)

        if direction == 'FORWARD/BACKWARD':
            remaining = set(new_indexs)
            selected = []
            result_dict = {}

            # the first model is the null model with only Intercept, or if start_from is provided, it the model with these variables
            if len(start_from) == 0:
                init_model = smf.glm(formula="y ~ 1", data=dataset, family=sm.families.Binomial(),freq_weights=sample_weight).fit()
            else:
                formula = "{} ~ {}".format('y', ' + '.join(new_start_from))
                init_model = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial(),freq_weights=sample_weight).fit()
                remaining = remaining - set(new_start_from)
                selected = new_start_from

            current_score, best_new_score = init_model.aic, init_model.aic
            reduced_loglikelihood = init_model.llf
            remaining = list(remaining)

            while current_score >= best_new_score:
                if remaining:
                    fwd_result = _forward(dataset, logging_file, current_score,
                                          best_new_score, remaining, selected,
                                          result_dict, reduced_loglikelihood,sample_weight)

                    if fwd_result != None:
                        current_score = fwd_result[0]
                        best_new_score = fwd_result[1]
                        result_dict = fwd_result[2]
                        selected = fwd_result[3]
                        remaining = fwd_result[4]
                        reduced_loglikelihood = fwd_result[5]
                    else:
                        logging_file.info("FORWARD completed and no variable selected to add")
                else:
                    fwd_result = None
                    logging_file.info("FORWARD has no more remaining variables to add")

                if len(selected) == 1:
                    continue
                elif len(selected) > 1:
                    bkd_result = _backward(dataset, logging_file, current_score,
                                          best_new_score, remaining, selected,
                                          result_dict, reduced_loglikelihood,sample_weight)

                    if bkd_result != None:
                        current_score = bkd_result[0]
                        best_new_score = bkd_result[1]
                        result_dict = bkd_result[2]
                        selected = bkd_result[3]
                        remaining = bkd_result[4]
                        reduced_loglikelihood = bkd_result[5]
                    else:
                        logging_file.info("BACKWARD completed and no variable selected to delete")
                else:
                    bkd_result = None
                    logging_file.info("BACKWARD has no more selected variables to delete")

                if not fwd_result and not bkd_result:
                    break




        elif direction == 'BACKWARD/FORWARD':
            remaining = []
            selected = list(set(new_indexs))
            result_dict = {}
            # the first model is the full model with all the variables
            formula = "{} ~ {}".format('y', ' + '.join(selected))
            init_model = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial(),freq_weights=sample_weight).fit()
            current_score, best_new_score = init_model.aic, init_model.aic
            reduced_loglikelihood = init_model.llf

            while current_score >= best_new_score:
                if len(selected) == 1:
                    continue
                elif len(selected) > 1:
                    bkd_result = _backward(dataset, logging_file, current_score,
                                          best_new_score, remaining, selected,
                                          result_dict, reduced_loglikelihood,sample_weight)

                    if bkd_result != None:
                        current_score = bkd_result[0]
                        best_new_score = bkd_result[1]
                        result_dict = bkd_result[2]
                        selected = bkd_result[3]
                        remaining = bkd_result[4]
                        reduced_loglikelihood = bkd_result[5]
                    else:
                        logging_file.info("BACKWARD completed and no variable selected to delete")
                else:
                    bkd_result = None
                    logging_file.info("BACKWARD has no more selected variables to delete")


                if remaining:
                    fwd_result = _forward(dataset, logging_file, current_score,
                                          best_new_score, remaining, selected,
                                          result_dict, reduced_loglikelihood,sample_weight)

                    if fwd_result != None:
                        current_score = fwd_result[0]
                        best_new_score = fwd_result[1]
                        result_dict = fwd_result[2]
                        selected = fwd_result[3]
                        remaining = fwd_result[4]
                        reduced_loglikelihood = fwd_result[5]
                    else:
                        logging_file.info("FORWARD completed and no variable selected to add")
                else:
                    fwd_result = None
                    logging_file.info("FORWARD has no more remaining variables to add")

                if not fwd_result and not bkd_result:
                    break


        # results
        result_aic = pd.DataFrame(result_dict).transpose()\
                   .reset_index()\
                   .rename(columns={'index': 'new_var_code'})

        if selected:
            formula = "{} ~ {}".format('y', ' + '.join(selected))
            model = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial(),freq_weights=sample_weight).fit()
            logging_file.info("##################### STEPWISE FINAL MODEL #####################################")
            logging_file.info(model.summary())
            pvalue = model.pvalues
            logging_file.info("Remove insignificant variables and those with negative coefficients")
            # 即使不显著，也要保留住start_from的变量
            pvalue = pvalue.drop(['Intercept'] + new_start_from)

            while pvalue.max() > 0.05 or sum(model.params.loc[model.params.index != 'Intercept'] < 0) > 0:
                while pvalue.max() > 0.05:
                    pvalue = pvalue.drop(pvalue.idxmax())
                    if len(pvalue) == 0:
                        break
                    formula = '{}~{}'.format('y', "+".join(pvalue.index))
                    model = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial(),freq_weights=sample_weight).fit()
                    pvalue = model.pvalues
                    pvalue = pvalue.drop('Intercept')
                    logging_file.info("Max p-values: %s, variable: %s" % (pvalue.max(), pvalue.idxmax()))

                while sum(model.params.loc[model.params.index != 'Intercept'] < 0) > 0:
                    coefs = model.params.loc[model.params.index != 'Intercept'].copy()
                    negative_coef_variables = list(coefs.loc[coefs < 0].index)
                    pvalue = model.pvalues
                    pvalue = pvalue.drop('Intercept')
                    # drop coef为负中pvalue最大的
                    negative_coef_pvalue = pvalue.loc[negative_coef_variables]
                    logging_file.info("Max p-values: %s, variable: %s" % (negative_coef_pvalue.max(), negative_coef_pvalue.idxmax()))
                    pvalue = pvalue.drop(negative_coef_pvalue.idxmax())
                    formula = '{}~{}'.format('y', "+".join(pvalue.index))
                    model = smf.glm(formula=formula, data=dataset, family=sm.families.Binomial()).fit()


            logging_file.info("SIGNIFICANT MODEL")
            logging_file.info(model.summary())

            final_selected = model.pvalues.drop('Intercept').index
        else:
            final_selected = selected

        result_aic = result_aic.loc[result_aic.new_var_code.isin(final_selected)]
        imp_df = pd.merge(mapping, result_aic, how='left', on='new_var_code')
        imp_df.loc[:, 'final_selected'] = np.where(imp_df.new_var_code.isin(final_selected), 1, 0)
        imp_df = imp_df.drop('new_var_code', axis=1)

        return imp_df


    def variable_clustering(self, X_cat, woe_iv_df, n_clusters=15):
        X_transformed = mt.BinWoe().transform_x_all(X_cat, woe_iv_df)
        agglo = FeatureAgglomeration(n_clusters=n_clusters)
        if len(X_transformed) > 20000:
            X_agglo = X_transformed.sample(20000)
        else:
            X_agglo = X_transformed.copy()
        agglo.fit(X_agglo)
        vars_clusters = pd.DataFrame(data={'指标英文':X_transformed.columns.tolist(),
                                           'cluster':list(agglo.labels_)})\
                          .sort_values('cluster')
        return vars_clusters, X_transformed


    def overall_ranking(self, X, y, var_dict, args_dict, methods, num_max_bins=10, n_clusters=15, verbose=True):
        """
        综合各算法的综合排序

        Args:
        X (pd.DataFrame): x 变量宽表, 已经处理过missing value的。如果没有，会处理。
        y (pd.Series): y label
        var_dict (pd.DataFrame): 变量字典表。必须包含：
            [u'数据源', u'指标英文', u'指标中文', u'数据类型', u'指标类型']
        args_dict (dict): key 是算法名称如['random_forest', 'svm', 'xgboost', 'IV', 'lasso']
            value是dict包含key值grid_search, param.如果没有赋值default值
        methods (list): ['random_forest', 'svm', 'xgboost', 'IV', 'lasso']
        num_max_bins (int): 用于自动分箱决定分箱的数量, default=10

        Returns:
        woe_iv_df (pd.DataFrame): 输出的auto classing的WOE IV表，可以直接贴到输出文档里面
        result (pd.DataFrame): 每个变量用每个方法计算的metrics和相应的ranking还有综合排序
        """
        num_missing = pd.isnull(X).sum().sum()
        if num_missing > 0:
            X = mu.process_missing(X,var_dict)

        result_df_list = []
        # IV. calculate_iv里面会分箱，所以需要传入的数据是原始的，只经过缺失处理的数据
        # 计算IV是mandatory
        bin_obj = mt.BinWoe()
        logging.info('Overall ranking prep: auto binning started')
        X_cat, all_encoding_map, all_spec_dict = bin_obj.binning(X, y, var_dict, num_max_bins, verbose=verbose)
        logging.info('Overall ranking prep: auto binning completed')
        woe_iv_df = bin_obj.calculate_woe_all(X_cat, y, var_dict, all_spec_dict, verbose=verbose)
        logging.info('Overall ranking prep: woe calculation completed')
        rebin_spec = bin_obj.create_rebin_spec(woe_iv_df, all_spec_dict, all_encoding_map)
        iv_result = woe_iv_df[['指标英文', 'IV']].copy().rename(columns={'指标英文':'var_code'}).drop_duplicates().dropna()
        iv_result.loc[:, 'iv_rank'] = iv_result.IV.rank(ascending=False)
        result_df_list.append(iv_result)
        logging.info('Overall ranking prep: IV completed')

        # 将categorical变量label encode，numerical保持原样
        def _encoding_categorical(data, var_dict):
            new_data = data.copy()
            if '数据类型' in var_dict.columns.values:
                cateogory_col = var_dict.loc[var_dict['指标英文'].isin(new_data.columns) & (var_dict['数据类型']=='varchar'), '指标英文'].tolist()
            else:
                cateogory_col = [col for col in new_data.columns if new_data[col].dtype == 'object']

            for col in cateogory_col:
                # NA不fill成missing的话，会一个NA一个label
                new_data.loc[:, col] = mu.label_encode(data[col])
            return new_data
        
        X = _encoding_categorical(X, var_dict)


        if 'random_forest' in methods:
            random_forest_result = self.random_forest(X, y, \
                                            args_dict['random_forest'].get('grid_search', False), \
                                            args_dict['random_forest'].get('param', None))
            result_df_list.append(random_forest_result)
            logging.info('Overall ranking prep: Random Forest completed')

        # 太慢了，SVC已经是基于libsvm的，速度没办法更快了
        if 'svm' in methods:
            svm_result = self.svm(X, y)
            result_df_list.append(svm_result)
            logging.info('Overall ranking prep: SVM completed')

        if 'xgboost' in methods:
            xgboost_result = self.xgboost(X, y,args_dict['xgboost'].get('grid_search', False),args_dict['xgboost'].get('param', None))
            result_df_list.append(xgboost_result)
            logging.info('Overall ranking prep: Xgboost completed')

        if 'lasso' in methods:
            lasso_result = self.lasso(X, y)
            result_df_list.append(lasso_result)
            logging.info('Overall ranking prep: LASSO completed')

        result = reduce(lambda left, right: left.merge(right, on='var_code', how='outer'),\
                        result_df_list)

        rank_columns = [i for i in result.columns if 'rank' in i]
        result.loc[:, 'overall_rank'] = (result[rank_columns] * 1.0).mean(1)
        result = var_dict.loc[:, ['指标英文', '指标中文','数据源']].drop_duplicates()\
                     .merge(result.rename(columns={'var_code': '指标英文'}), \
                            on='指标英文')

        cluster_result, X_transformed = self.variable_clustering(X_cat, woe_iv_df, n_clusters=n_clusters)
        result = result.merge(cluster_result, on='指标英文', how='left')

        return X_cat, X_transformed, woe_iv_df, rebin_spec, result

# add overall_ranking for sample_weight

    def overall_ranking_sw(self, X, y,sample_weight, var_dict, args_dict, methods, num_max_bins=10, n_clusters=15, verbose=True):
        """
        综合各算法的综合排序

        Args:
        X (pd.DataFrame): x 变量宽表, 已经处理过missing value的。如果没有，会处理。
        y (pd.Series): y label
        sample_weight(pd.Series):样本权重
        var_dict (pd.DataFrame): 变量字典表。必须包含：
            [u'数据源', u'指标英文', u'指标中文', u'数据类型', u'指标类型']
        args_dict (dict): key 是算法名称如['random_forest', 'svm', 'xgboost', 'IV', 'lasso']
            value是dict包含key值grid_search, param.如果没有赋值default值
        methods (list): ['random_forest', 'svm', 'xgboost', 'IV', 'lasso']
        num_max_bins (int): 用于自动分箱决定分箱的数量, default=10

        Returns:
        woe_iv_df (pd.DataFrame): 输出的auto classing的WOE IV表，可以直接贴到输出文档里面
        result (pd.DataFrame): 每个变量用每个方法计算的metrics和相应的ranking还有综合排序
        """
        num_missing = pd.isnull(X).sum().sum()
        if num_missing > 0:
            X = mu.process_missing(X,var_dict)

        result_df_list = []
        # IV. calculate_iv里面会分箱，所以需要传入的数据是原始的，只经过缺失处理的数据
        # 计算IV是mandatory
        bin_obj = mt.BinWoe()
        logging.info('Overall ranking prep: auto binning started')
        X_cat, all_encoding_map, all_spec_dict = bin_obj.binning(X, y, var_dict, num_max_bins, verbose=verbose)
        logging.info('Overall ranking prep: auto binning completed')
        woe_iv_df = bin_obj.calculate_woe_all_sw(X_cat, y,sample_weight, var_dict, all_spec_dict, verbose=verbose)
        logging.info('Overall ranking prep: woe calculation completed')
        rebin_spec = bin_obj.create_rebin_spec(woe_iv_df, all_spec_dict, all_encoding_map)
        iv_result = woe_iv_df[['指标英文', 'IV']].copy()\
                            .rename(columns={'指标英文':'var_code'})\
                            .drop_duplicates().dropna()
        iv_result.loc[:, 'iv_rank'] = iv_result.IV.rank(ascending=False)
        result_df_list.append(iv_result)
        logging.info('Overall ranking prep: IV completed')

        # 将categorical变量label encode，numerical保持原样
        def _encoding_categorical(data, var_dict):
            new_data = data.copy()
            if '数据类型' in var_dict.columns.values:
                cateogory_col = var_dict.loc[var_dict['指标英文'].isin(new_data.columns) &\
                                                  (var_dict['数据类型']=='varchar'), '指标英文'].tolist()
            else:
                cateogory_col = [col for col in new_data.columns if new_data[col].dtype == 'object']

            for col in cateogory_col:
                # NA不fill成missing的话，会一个NA一个label
                new_data.loc[:, col] = mu.label_encode(data[col])
            return new_data

        X = _encoding_categorical(X, var_dict)


        if 'random_forest' in methods:
            #print(type(sample_weight))
            #print('randomforest')
            random_forest_result = self.random_forest_sw(X, y,sample_weight, \
                                            args_dict['random_forest'].get('grid_search', False), \
                                            args_dict['random_forest'].get('param', None))
            result_df_list.append(random_forest_result)
            logging.info('Overall ranking prep: Random Forest completed')

        # 太慢了，SVC已经是基于libsvm的，速度没办法更快了
        if 'svm' in methods:
            svm_result = self.svm_sw(X, y,sample_weight)
            result_df_list.append(svm_result)
            logging.info('Overall ranking prep: SVM completed')

        if 'xgboost' in methods:
            #print(sample_weight)
            xgboost_result = self.xgboost_sw(X, y,sample_weight,\
                                    args_dict['xgboost'].get('grid_search', False), \
                                    args_dict['xgboost'].get('param', None))
            result_df_list.append(xgboost_result)
            logging.info('Overall ranking prep: Xgboost completed')

        if 'lasso' in methods:
            lasso_result = self.lasso(X, y)
            result_df_list.append(lasso_result)
            logging.info('Overall ranking prep: LASSO completed')

        result = reduce(lambda left, right: left.merge(right, on='var_code', how='outer'),\
                        result_df_list)

        rank_columns = [i for i in result.columns if 'rank' in i]
        result.loc[:, 'overall_rank'] = (result[rank_columns] * 1.0).mean(1)
        result = var_dict.loc[:, ['指标英文', '指标中文','数据源']].drop_duplicates()\
                     .merge(result.rename(columns={'var_code': '指标英文'}), \
                            on='指标英文')

        cluster_result, X_transformed = self.variable_clustering(X_cat, woe_iv_df, n_clusters=n_clusters)
        result = result.merge(cluster_result, on='指标英文', how='left')

        return X_cat, X_transformed, woe_iv_df, rebin_spec, result

    def select_from_ranking(self, overall_ranking_cl, total_topn, source_topn, cluster_topn, iv_threshold):
        """
        变量选择：
        1. 各数据源排名靠前的
        2. 将变量聚类，各类型排名靠前的
        3. IV取值高于一个阈值
        4. 综合排名靠前的

        Args:
        overall_ranking_cl (pd.DataFrame): self.overall_ranking()返回的排序结果
            merge上cluster分类的结果
        total_topn (int): top n kept based on overall ranking
        source_topn (int): top n kept within each data source based on overall ranking
        cluster_topn (int): top n kept within each claster based on overal ranking
        iv_threshold (float): the minimum value of iv for variables to be kept
        """
        total_kept = overall_ranking_cl.sort_values('overall_rank')[u'指标英文']\
                                       .iloc[:total_topn].tolist()
        cluster_kept = overall_ranking_cl.sort_values(['cluster','overall_rank'])\
                                         .groupby('cluster')\
                                         .apply(lambda x:x.head(cluster_topn))\
                                         [u'指标英文'].tolist()
        source_kept = overall_ranking_cl.sort_values(['数据源','overall_rank'])\
                                         .groupby('数据源')\
                                         .apply(lambda x:x.head(source_topn))\
                                         [u'指标英文'].tolist()
        iv_kept = overall_ranking_cl.loc[overall_ranking_cl.IV>iv_threshold, '指标英文'].tolist()
        kept_vars = list(set(total_kept) | set(cluster_kept) | set(source_kept) | set(iv_kept))
        return kept_vars


class Colinearity(object):
    def __init__(self, X, RESULT_PATH):
        self.X = X
        self.RESULT_PATH = RESULT_PATH

    def run(self, ranking_result, corr_threshold=0.7, vif_threshold=10):
        self.calculate_corr(ranking_result, corr_threshold)
        self.X = self.X.drop(self.corr_exclude_vars, 1)
        self.calculate_vif(vif_threshold)


    def calculate_corr(self, ranking_result, corr_threshold=0.7):
        corr_df = self.X.corr()
        pl.plot_colinearity(corr_df, self.RESULT_PATH, 'OverallRankingKept')

        corr_df = corr_df.stack().reset_index()
        corr_df.columns = ['var1', 'var2', 'correlation']
        corr_df.loc[:, 'flipped_var1'] = corr_df.var2
        corr_df.loc[:, 'flipped_var2'] = corr_df.var1
        df1 = corr_df[['var1', 'var2']].reset_index()
        df2 = corr_df[['flipped_var1', 'flipped_var2']].reset_index()
        df = df1.merge(df2, left_on=['var1', 'var2'], right_on=['flipped_var1', 'flipped_var2'])
        df.loc[:, 'indice_list'] = df.apply(lambda x: json.dumps(sorted([x['index_x'], x['index_y']])), 1)
        kept_index = df.drop_duplicates('indice_list').loc[df['index_x'] != df['index_y'], 'index_x']
        corr_df = corr_df.loc[kept_index, ['var1', 'var2', 'correlation']]
        corr_df.loc[:, 'abs_corr'] = abs(corr_df.correlation)

        ranking_result = ranking_result.rename(columns={'指标英文':'var1'})
        self.corr_df = corr_df.merge(ranking_result[['var1', 'IV']], on='var1', how='left')\
                                   .rename(columns={'IV': 'var1_IV'})
        ranking_result = ranking_result.rename(columns={'var1':'var2'})
        self.corr_df = self.corr_df.merge(ranking_result[['var2', 'IV']], on='var2', how='left')\
                                   .rename(columns={'IV': 'var2_IV'})

        self.corr_df = self.corr_df.sort_values('abs_corr', ascending=False)
        over_threhold = self.corr_df.loc[self.corr_df.correlation >= corr_threshold].copy()
        self.corr_exclude_vars = []
        for i in range(len(over_threhold)):
            the_pair = over_threhold.iloc[i]
            # corr超标的时候，保留IV更大的，但是如果要保留的在更高corr的一个pair中因IV更低所以被删除了
            # 则这个pair里面的IV较小的不会被exclude
            if the_pair['var1'] not in self.corr_exclude_vars and \
                the_pair['var2'] not in self.corr_exclude_vars:

                if the_pair['var1_IV'] >= the_pair['var2_IV']:
                    self.corr_exclude_vars.append(the_pair['var2'])
                else:
                    self.corr_exclude_vars.append(the_pair['var1'])

        self.corr_exclude_vars = list(set(self.corr_exclude_vars))
        #final correlation
        corr_df2 = self.X.iloc[:,~self.X.columns.isin(self.corr_exclude_vars)].corr().stack().reset_index()
        corr_df2.columns = ['var1', 'var2', 'correlation']
        self.final_corr_df = corr_df2[corr_df2.var1 != corr_df2.var2]

    def calculate_vif(self, vif_threshold=10):
        """
        每删除VIF最高的一个变量，余下的变量重新计算VIF并根据vif_threshold删除
        """
        self.keepvars = self.X.columns.tolist()
        self.vif_dropvars = []
        dropped = True
        # 当前vif
        vif0 = [variance_inflation_factor(np.array(self.X), ix) for ix in range(len(self.keepvars))]
        firstvif = pd.DataFrame({"指标英文": self.keepvars, "VIF": vif0})
        self.firstvif = firstvif.sort_values('VIF', ascending=False)
        # 删除超过vif阈值的变量
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(self.X[self.keepvars].values, ix) for ix in range(len(self.keepvars))]
            maxloc = vif.index(max(vif))
            if max(vif) >= vif_threshold:
                logging.info(("dropping '" + self.X[self.keepvars].columns[maxloc] + "' at index: " + str(maxloc)))
                self.vif_dropvars.append(self.keepvars[maxloc])
                del self.keepvars[maxloc]
                dropped = True
        # 删除变量后的vif
        lastvif = pd.DataFrame({"指标英文": self.keepvars, "VIF": vif})
        self.lastvif = lastvif.sort_values('VIF', ascending=False)
