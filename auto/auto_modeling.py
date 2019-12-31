# encoding=utf-8
"""
Python 3.6

自动建模，传统评分卡模型
"""
import os
import sys
import logging
import codecs
import argparse
from functools import reduce
from imp import reload
from datetime import datetime, timedelta
from copy import deepcopy

import numpy as np
import pandas as pd
from jinja2 import Template
from sklearn.model_selection import train_test_split, KFold
try:
    import xgboost as xgb
except:
    pass

import utils3.plotting as pl
import utils3.misc_utils as mu
import utils3.metrics as mt
import utils3.summary_statistics as ss
import utils3.feature_selection as fs
import utils3.modeling as ml
import utils3.verify_performance as vp
from utils3.data_io_utils import *



"""
The ultimate params

args_dict = {
    'random_forest': {
        'grid_search': False,
        'param': None
    },
    'xgboost': {
        'grid_search': False,
        'param': None
    }
}
methods = [
    'random_forest',
    'lasso',
    # 'xgboost'
]


params = {
'all_x_y': pd.DataFrame(),
'var_dict': pd.DataFrame(),
'x_cols': [],
'y_col': '',
'useless_vars': [],
'exempt_vars': [],
'ranking_args_dict': args_dict,
'ranking_methods': methods,
#
'DATA_PATH': '',
'RESULT_PATH': '',
'SAVE_LABEL': '',
'TIME_COL': '',
'TRAIN_SET_NAME': '',
'TEST_SET_NAME': '',
'OOT_SET_NAME': '',
# 可填
'NFOLD': 5,
'uniqvalue_cutoff': 0.97,
'missingrate_cutoff': 0.75,
'badrate_cutoff': 0.75,
'xgbparams_selection_method': xgbparams_selection_method,
'KEPT_LIMIT': 50,
'xgb_params_range': None,
'xgb_param_experience': None,
}

必选 Arguments:
- all_x_y (pd.DataFrame()): 建模整理好的数据，缺失值已经处理好，通常包括每单申请的一些基本信息
    （如product_name, gamegroup等），Y和X数据。另外，index必须为区分申请的unique id
- var_dict (pd.DataFrame()): 标准变量字典表，包含以下这些列：数据源，数据类型，指标类型，
指标英文，指标中文。
- x_cols (list): X变量名list
- y_col (str): column name of y
- useless_vars (list): 跑EDA summary时已知的无用变量名list，在exclusion reason里会被列为无用变量
- exempt_vars (list): 跑EDA summary时豁免变量名list，这些变量即使一开始被既定原因定为exclude，也会被
    保留，比如前一版本模型的变量
- ranking_args_dict (dict): key 是算法名称如['random_forest', 'svm', 'xgboost', 'IV', 'lasso']
    value是dict包含key值grid_search, param.如果没有赋值default值
- ranking_methods (list): ['random_forest', 'svm', 'xgboost', 'IV', 'lasso']


- TRAIN_SET_NAME (str): all_x_y中'sample_set'列取值为train set的具体取值
- TEST_SET_NAME (str): all_x_y中'sample_set'列取值为test set的具体取值, 如果为空则认为没有test set的需要
- DATA_PATH (str): 数据存储路径
- RESULT_PATH (str): 结果存储路径
- SAVE_LABEL (str): summary文档存储将用'%s_variables_summary.xlsx' % SAVE_LABEL
- TIME_COL (str): the column name indicating "time". e.g. 'apply_month', 'applicationdate'

可选 Keyword Arguments:
- uniqvalue_cutoff（float): 0-1之间，跑EDA summary表格时的缺失率和唯一值阈值设定。default=0.97
- missingrate_cutoff (float): 每个变量会计算按照时间维度的缺失率的std，这些std会被排序，排名前百分之几的会保留。defaut=0.75
- badrate_cutoff (float):
- xgbparams_selection_method (list): ['XGBExp', 'XGBGrid', 'XGBRandom'],如果有xgboost模型的时候
- xgb_params_range (dict): xgboost gridsearch/randomsearch的搜索范围区间
    e.g. xgb_params_range = {
            'learning_rate': [0.05,0.1,0.2],
            'max_depth': [2, 3],
            'gamma': [0, 0.1],
            'min_child_weight':[1,3,10,20],
            # 'subsample': np.linspace(0.4, 0.7, 3),
            # 'colsample_bytree': np.linspace(0.4, 0.7, 3),
        }
- xgb_param_experience (dict): xgboost自定义经验参数
    e.g. xgb_param_experience = {
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
- KEPT_LIMIT (int): 变量选择最终希望保留的数量。default=50


params_verify = {
'data_cat_dict': {},
'xgboost_data_dict': {},
'var_dict': pd.DataFrame(),
'all_Y': pd.DataFrame(),
'y_cols': [],
'union_var_list': [],
'model_labels': [],
'coarse_classing_rebin_spec': {},
'liftcurve_name_map': {},
#
'TRAIN_SET_NAME': '',
'TEST_SET_NAME': '',
'OOT_SET_NAME': '',
'BACKSCORE_SET_NAME': '',
'DATA_PATH': '',
'RESULT_PATH': '',
'Y_BUILD_NAME': '',
#可选
'backscore_has_perf': [],
'base_model_score': {},
'BASE_MODEL_LABEL': ''
}

必选：
- data_cat_dict (dict): key为sample_set的名称。value为分箱好的数据，index为
    区分每条数据的unique identifier
- xgboost_data_dict (dict): key为sample_set的名称。value为xgboost要用的数据，
    index为区分每条数据的unique identifier。
- var_dict (pd.DataFrame()): 标准变量字典表，包含以下这些列：数据源，数据类型，指标类型，
指标英文，指标中文。
- all_Y (pd.DataFrame): 所有数据的(TRAIN, TEST, OOT etc)的Y，不同表现期的Y
    index为区分每条数据的unique identifier
- y_cols (list): all_Y中需要用于对比表现的Y的column name
- union_var_list (list): 所有需要对比的模型所选中的变量，只限逻辑回归模型
- model_labels (list): 需要对比的模型model_label名称
- coarse_classing_rebin_spec (dict): 粗分箱spec
- liftcurve_name_map (dict): 画lift curve的时候原始model_label转换成图中的
    显示名称，因为画图会将中文显示成方格，所以转换的值需全为英文。取名时请注意规范一致
    性和可读性和理解性，因为这个是会放到最终报告中的。key值为建模时各模型对应的model_label,
    value值为规范刻度和解释性较好的全英文正式模型名称



- TRAIN_SET_NAME (str): all_x_y中'sample_set'列取值为train set的具体取值
- TEST_SET_NAME (str): all_x_y中'sample_set'列取值为test set的具体取值, 如果为空则认为没有test set的需要
- OOT_SET_NAME (str): all_x_y中'sample_set'列取值为OOT set的具体取值, 如果为空则认为没有oot set
- BACKSCORE_SET_NAME (str): 'sample_set'列取值为backscore set的具体取值
- DATA_PATH (str): 数据存储路径
- RESULT_PATH (str): 结果存储路径
- Y_BUILD_NAME (str): 建模所用的Y的列明


可选：
- backscore_has_perf (list): list of applyid (unique identifier)用于区分
    backscore样本中有表现的样本。default=[]
- base_model_score (dict): 用于对比的老模型的分数，key为sample_set的名称。
    value为分数，index为区分每条数据的unique identifier。存储形式为pd.Series。
    如果需要与老模型，或者有老模型做为对比时，此为必选
- BASE_MODEL_LABEL (str): 用于对比的老模型或者benchmark模型正式的模型名称。如果需要
    与老模型，或者有老模型做为对比时，此为必选
"""

class StepOneEDA(object):
    """
    第一步进行数据EDA，会产生EDA summary 表格，并根据每个月，或者所选中的时间维度对比缺失率是否稳定
    ，逾期率是否稳定
    """
    def __init__(self, params):
        """
        Args:
        params (dict): dict 包含以下
        - all_x_y (pd.DataFrame()): 建模整理好的数据，缺失值已经处理好，通常包括每单申请的一些基本信息
            （如product_name, gamegroup等），Y和X数据。另外，index必须为区分申请的unique id，
            也必须包含columns:'sample_set', 用于区分train, test
        - var_dict (pd.DataFrame()): 标准变量字典表，包含以下这些列：数据源，数据类型，指标类型，
        指标英文，指标中文。
        - x_cols (list): X变量名list
        - y_col (str): column name of y
        - useless_vars (list): 跑EDA summary时已知的无用变量名list，在exclusion reason里会被列为无用变量
        - exempt_vars (list): 跑EDA summary时豁免变量名list，这些变量即使一开始被既定原因定为exclude，也会被
            保留，比如前一版本模型的变量
        - RESULT_PATH (str): 结果存储路径
        - SAVE_LABEL (str): summary文档存储将用'%s_variables_summary.xlsx' % SAVE_LABEL
        - TIME_COL (str): the column name indicating "time". e.g. 'apply_month', 'applicationdate'
        - uniqvalue_cutoff（float): 0-1之间，跑EDA summary表格时的缺失率和唯一值阈值设定。default=0.97
        - missingrate_cutoff (float): 每个变量会计算按照时间维度的缺失率的std，这些std会被排序，排名前百分之几的会保留。defaut=0.75
        - badrate_cutoff (float): 每个变量会计算按照时间维度的逾期率排序的std，这些std会被排序，排名前百分之几的会保留。defaut=0.75


        Returns:
        os.path.join(RESULT_PATH, '%s_variables_summary.xlsx' % SAVE_LABEL)为存储输出结果

        Examples:
        step1_obj = StepOneEDA(params)
        step1_obj.run()
        """
        self.all_x_y = params['all_x_y']
        if self.all_x_y.index.values[0] == 0:
            print("First index value = 0 discovered, please make sure that the index is set to be the unique identifier, e.g. apply_id")
            raise
        if len(np.unique(self.all_x_y.index.values)) != len(self.all_x_y.index.values):
            print("Duplicated indices discovered, please make sure there are no duplicates")
            raise

        self.var_dict = params['var_dict']
        self.x_cols = params['x_cols']
        self.y_col = params['y_col']
        self.useless_vars = params['useless_vars']
        self.exempt_vars = params['exempt_vars']
        self.uniqvalue_cutoff = params.get('uniqvalue_cutoff', 0.97)
        self.missingrate_cutoff = params.get('missingrate_cutoff', 0.75)
        self.badrate_cutoff = params.get('badrate_cutoff', 0.75)
        self.X = self.all_x_y[self.x_cols].copy()
        self.RESULT_PATH = params['RESULT_PATH']
        self.SAVE_LABEL = params['SAVE_LABEL']
        self.TIME_COL = params['TIME_COL']



    def run(self):
        logging.info("""第一步进行数据EDA，会产生EDA summary 表格，并根据每个月，或者所选中的时间维度对比缺失率是否稳定
          ，逾期率是否稳定。请确保输出的EDA summary表正确输出，并且已经根据EDA summary
            检查过数据异常并做了相关数据修正，同时相应的"exclusion_reason"那一列需要补充填写的都已
            填写完毕。所有"exclusion_reason"为空的指标都是在EDA阶段后保留的""")

        logging.info("EDA 表格输出: 所有set以及各set单独分别输出EDA表格")

        output_file_names = []
        for sample_set in self.all_x_y.sample_set.unique():
            apply_list = self.all_x_y.loc[self.all_x_y.sample_set==sample_set].index.values
            ss.eda(self.X.loc[apply_list], self.var_dict, self.useless_vars, self.exempt_vars, self.RESULT_PATH, self.SAVE_LABEL+sample_set, self.uniqvalue_cutoff)
            output_file_names.append(self.SAVE_LABEL+sample_set+'_variables_summary.xlsx')


        ss.eda(self.X, self.var_dict, self.useless_vars, self.exempt_vars, self.RESULT_PATH, self.SAVE_LABEL, self.uniqvalue_cutoff)

        logging.info("按时间维度画缺失率图: 因不同的set可能跨越不同的时间段，因此是所有set的数据综合一起按照时间维度画图")
        self.missing_rates_for_all()

        logging.info("按时间维度画逾期率趋势图: 因不同的set可能跨越不同的时间段，因此是所有set的数据综合一起按照时间维度画图")
        self.compare_bad_rate()

        self.summary_eda_table.to_excel(os.path.join(self.RESULT_PATH, '%s_variables_summary.xlsx' % self.SAVE_LABEL), index=False)

        logging.info("""
        本阶段输出文件：
          1. `RESULT_PATH`路径下『'%s_variables_summary.xlsx'』文件。内容包括所有set的数据的EDA表格和相应的筛除原因等。
          2. `RESULT_PATH`路径下 %s 文件。内容包括各自数据集对应的数据的EDA表格和相应的筛除原因等。
          3. `RESULT_PATH`路径下 `figure/badByTime`（按时间对比逾期率）和`figure/missingByTime`（按时间对比缺失率）文件夹, 文件夹内包含画图

        """ % (self.SAVE_LABEL, json.dumps(output_file_names)))


    def missing_rates_for_all(self):
        """
        eda summary表格保留的字段，会按照TIME_COL时间维度检查缺失率，画图，并统计趋势率之间的
        """
        self.summary_eda_table = pd.read_excel(os.path.join(self.RESULT_PATH, '%s_variables_summary.xlsx' % self.SAVE_LABEL))
        self.eda_kept_cols = self.summary_eda_table.loc[self.summary_eda_table.exclusion_reason.isnull(), '指标英文'].unique()
        variable_missing_std = {}
        for col in self.eda_kept_cols:
            logging.log(18, col + ' starts missing rate by time calculation and plotting')
            rates_by_month, plt = pl.missing_trend(self.all_x_y[col], self.all_x_y[self.TIME_COL], self.RESULT_PATH)
            plt.close()
            variable_missing_std[col] = np.nanstd(rates_by_month)

        all_rates_by_month = pd.Series(variable_missing_std).sort_values(ascending=True) # 越小越好，排名高
        keep_n = int(np.floor(len(self.eda_kept_cols) * self.missingrate_cutoff))
        missing_rate_to_exclude = all_rates_by_month.iloc[keep_n:].index.values
        self.summary_eda_table.loc[self.summary_eda_table['指标英文'].isin(missing_rate_to_exclude), 'exclusion_reason'] = '按时间维度统计缺失率变化过高去除'
        all_rates_by_month = all_rates_by_month.to_frame('按时间维度统计缺失率STD').reset_index().rename(columns={'index':'指标英文'})
        col_orders = list(self.summary_eda_table.columns) + ['按时间维度统计缺失率STD']
        self.summary_eda_table = self.summary_eda_table.merge(all_rates_by_month, on='指标英文', how='left')
        self.summary_eda_table = self.summary_eda_table[col_orders]




    def compare_bad_rate(self):
        X_cat10, _, _ = mt.BinWoe().binning(self.X[self.eda_kept_cols], self.all_x_y[self.y_col], self.var_dict, 10)
        X_cat10_with_time = X_cat10.merge(self.all_x_y[[self.TIME_COL, self.y_col]], left_index=True, right_index=True)
        eventrate_by_time = X_cat10_with_time[[self.TIME_COL, self.y_col]].groupby(self.TIME_COL)[self.y_col].mean()

        bad_rates_change_metrics = {}
        for col in self.eda_kept_cols:
            logging.log(18, col + ' starts event rate by time calculation and plotting')
            eventrate_time_pivot = X_cat10_with_time[[col, self.TIME_COL, self.y_col]]\
                                            .pivot_table(index=col,
                                                         columns=self.TIME_COL,
                                                         values=self.y_col)
            eventrate_time_pivot_rank = eventrate_time_pivot.rank()

            pl.check_badrate_trend_simplified(eventrate_time_pivot,self.RESULT_PATH,
                                            eventrate_by_time=eventrate_by_time,
                                            heatmap=True)

            bad_rates_change_metrics[col] = eventrate_time_pivot_rank.std(1, skipna=True).mean(skipna=True)

        bad_rates_by_time = pd.Series(bad_rates_change_metrics).sort_values(ascending=True) # 越小越好，排名高
        keep_n = int(np.floor(len(self.eda_kept_cols) * self.badrate_cutoff))
        bad_rate_to_exclude = bad_rates_by_time.iloc[keep_n:].index.values
        self.summary_eda_table.loc[self.summary_eda_table['指标英文'].isin(bad_rate_to_exclude), 'exclusion_reason'] = '按时间维度统计逾期率排序变化过高去除'
        bad_rates_by_time = bad_rates_by_time.to_frame('按时间维度统计逾期率排序STD').reset_index().rename(columns={'index':'指标英文'})
        col_orders = list(self.summary_eda_table.columns) + ['按时间维度统计逾期率排序STD']
        self.summary_eda_table = self.summary_eda_table.merge(bad_rates_by_time, on='指标英文', how='left')
        self.summary_eda_table = self.summary_eda_table[col_orders]



class StepTwoVarSelect(object):
    """
    第二步进行变量选择，请确保第一步输出的EDA summary表正确输出，并且已经根据EDA summary
    检查过数据异常并做了相关数据修正，同时相应的"exclusion_reason"那一列需要补充填写的都已
    填写完毕。所有"exclusion_reason"为空的指标都是在EDA阶段后保留的
    """
    def __init__(self, params):
        """
        Args:
        params (dict): dict 包含以下
        - all_x_y (pd.DataFrame()): 建模整理好的数据，缺失值已经处理好，通常包括每单申请的一些基本信息
            （如product_name, gamegroup, sample_set等），Y和X数据。另外，index必须为区分申请的unique id
            也必须包含columns:'sample_set', 用于区分train, test
        - var_dict (pd.DataFrame()): 标准变量字典表，包含以下这些列：数据源，数据类型，指标类型，
        指标英文，指标中文。
        - x_cols (list): X变量名list
        - y_col (str): column name of y
        - ranking_args_dict (dict): key 是算法名称如['random_forest', 'svm', 'xgboost', 'IV', 'lasso']
            value是dict包含key值grid_search, param.如果没有赋值default值
        - ranking_methods (list): ['random_forest', 'svm', 'xgboost', 'IV', 'lasso']

        - TRAIN_SET_NAME (str): all_x_y中'sample_set'列取值为train set的具体取值
        - TEST_SET_NAME (str): all_x_y中'sample_set'列取值为test set的具体取值, 如果为空则认为没有test set的需要
        - DATA_PATH (str): 数据存储路径
        - RESULT_PATH (str): 结果存储路径
        - SAVE_LABEL (str): summary文档存储将用'%s_variables_summary.xlsx' % SAVE_LABEL
        - TIME_COL (str): the column name indicating "time". e.g. 'apply_month', 'applicationdate'
        - KEPT_LIMIT (int): 变量选择最终希望保留的数量。default=50


        Returns:
        save_data_to_json(cols_filter, self.RESULT_PATH, 'variable_filter.json') 包含每个步骤筛选涉及到的变量名list
        self.summary_eda_table.to_excel(os.path.join(self.RESULT_PATH, '%s_variables_summary.xlsx' % self.SAVE_LABEL), index=False)

        Examples:
        step2_obj = StepTwoVarSelect(params)
        step2_obj.variable_ranking(bin20=True)
        step2_obj.variable_filter(total_topn=50, source_topn=5, cluster_topn=5,
                                  iv_threshold=0.3, corr_threshold=0.7,
                                  vif_threshold=10)

        """
        self.all_x_y = params['all_x_y']
        if self.all_x_y.index.values[0] == 0:
            print("First index value = 0 discovered, please make sure that \
               the index is set to be the unique identifier, e.g. apply_id")
            raise
        if len(np.unique(self.all_x_y.index.values)) != len(self.all_x_y.index.values):
            print("Duplicated indices discovered, please make sure there \
                  are no duplicates")
            raise

        if 'sample_set' not in self.all_x_y.columns:
            print("'sample_set' column is not found in all_x_y, if no test \
                 set is required for the modeling, please set arg TEST=False")
            raise

        self.var_dict = params['var_dict']
        self.x_cols = params['x_cols']
        self.y_col = params['y_col']
        self.TRAIN_SET_NAME = params['TRAIN_SET_NAME']
        self.TEST_SET_NAME = params.get('TEST_SET_NAME', '')
        self.DATA_PATH = params['DATA_PATH']
        self.RESULT_PATH = params['RESULT_PATH']
        self.TIME_COL = params['TIME_COL']
        self.SAVE_LABEL = params['SAVE_LABEL']
        self.KEPT_LIMIT = params.get('KEPT_LIMIT', 50)

        self.X_train = self.all_x_y.loc[self.all_x_y.sample_set==self.TRAIN_SET_NAME, self.x_cols].copy()
        self.y_train = self.all_x_y.loc[self.all_x_y.sample_set==self.TRAIN_SET_NAME, self.y_col].copy()

        if self.TEST_SET_NAME:
            self.X_test = self.all_x_y.loc[self.all_x_y.sample_set==self.TEST_SET_NAME, self.x_cols].copy()
            self.y_test = self.all_x_y.loc[self.all_x_y.sample_set==self.TEST_SET_NAME, self.y_col].copy()

        self.summary_eda_table = pd.read_excel(os.path.join(self.RESULT_PATH, '%s_variables_summary.xlsx' % self.SAVE_LABEL))
        self.eda_exclude_cols = self.summary_eda_table.loc[self.summary_eda_table.exclusion_reason.notnull(), '指标英文']

        if 'ranking_args_dict' not in params:
            self.ranking_args_dict = {
                'random_forest': {
                    'grid_search': False,
                    'param': None
                },
                'xgboost': {
                    'grid_search': False,
                    'param': None
                }
            }
        else:
            self.ranking_args_dict = params['ranking_args_dict']

        if 'ranking_methods' not in params:
            ranking_methods = [
                'random_forest',
                # 'lasso',
                # 'xgboost'
            ]
        else:
            self.ranking_methods = params['ranking_methods']


    def variable_ranking(self, bin20=True):
        """
        Args:

        bin20 (bool): defaut=True, will additionally run auto classing 20bin
        """
        logging.info("第二步进行变量选择: 变量综合排序（根据所选方法如：random forest, lasso, xgboost, IV等排序）。所用数据集为：TRAIN")
        self.X_train = self.X_train.drop(self.eda_exclude_cols, 1)
        fs_obj = fs.FeatureSelection()
        X_cat_train, self.X_transformed_train, woe_iv_df, rebin_spec, ranking_result = \
                        fs_obj.overall_ranking(self.X_train, self.y_train, \
                                               self.var_dict, self.ranking_args_dict, \
                                               self.ranking_methods, \
                                               num_max_bins=10, n_clusters=5)
        ranking_result.to_excel(os.path.join(self.RESULT_PATH, 'overall_ranking.xlsx'))

        save_data_to_pickle({'X_cat_train':X_cat_train,
            'X_transformed_train': self.X_transformed_train,
            'rebin_spec': rebin_spec}, self.DATA_PATH, '建模细分箱10结果.pkl')

        woe_iv_df.to_excel(os.path.join(self.RESULT_PATH, 'woe_iv_df_auto_classing.xlsx'))
        if bin20:
            logging.info('start 20bin auto classing')
            X_cat, all_encoding_map, all_spec_dict = mt.BinWoe().binning(self.X_train, self.y_train, self.var_dict, 20)
            woe_iv_df20 = mt.BinWoe().calculate_woe_all(X_cat, self.y_train, self.var_dict, all_spec_dict)
            woe_iv_df20.to_excel(os.path.join(self.RESULT_PATH, 'woe_iv_df_auto_classing_20bin.xlsx'), index=False)
            logging.info("第二步进行变量选择: 变量综合排序。TRAIN数据集20等分自动分箱数据整理的WOE-IV等信息文档存储于：%s" % os.path.join(self.RESULT_PATH, 'woe_iv_df_auto_classing_20bin.xlsx'))

        logging.info("第二步进行变量选择: 变量综合排序。各算法排序结果存储文件：%s" % os.path.join(self.RESULT_PATH, 'overall_ranking.xlsx'))
        logging.info("第二步进行变量选择: 变量综合排序。TRAIN数据集自动分箱数据、WOE转换数据和自动分箱边界信息存储于：%s" % os.path.join(self.DATA_PATH, '建模细分箱10结果.pkl'))
        logging.info("第二步进行变量选择: 变量综合排序。TRAIN数据集10等分自动分箱数据整理的WOE-IV等信息文档存储于：%s" % os.path.join(self.RESULT_PATH, 'woe_iv_df_auto_classing.xlsx'))




    def variable_filter(self, total_topn=50, source_topn=5, cluster_topn=5,
                              iv_threshold=0.3, corr_threshold=0.7,
                              vif_threshold=10):
        """
        根据variable_ranking出的结果对变量进行筛选, 并对变量做共线性筛选

        Args:
        total_topn (int): default=50, top n kept based on overall ranking
        source_topn (int): default=5, top n kept within each data source based on overall ranking
        cluster_topn (int): default=5, top n kept within each claster based on overal ranking
        iv_threshold (float): deafult=0.3, the top n% to keep
        corr_threshold (float): default=0.7, correlation coefficient threshold, above it will be excluded
        vif_threshold (float): deafult=10, VIF threshold, above it will be excluded

        """
        logging.info("第二步进行变量选择: 变量筛选。根据综合排序进行变量筛选和共线性筛查。所用数据集为：TRAIN")

        ranking_result = pd.read_excel(os.path.join(self.RESULT_PATH, 'overall_ranking.xlsx'), index=False)

        selected = fs.FeatureSelection().select_from_ranking(ranking_result, total_topn, \
                                              source_topn, cluster_topn, \
                                              iv_threshold)

        logging.info("第二步进行变量选择: 变量筛选。根据Overall ranking选择的变量数量: %s" % len(selected))

        logging.info('第二步进行变量选择: 变量筛选。starts colinearity check')

        corr_obj = fs.Colinearity(self.X_transformed_train[selected].copy(), self.RESULT_PATH)
        corr_obj.run(ranking_result, corr_threshold, vif_threshold)

        writer = pd.ExcelWriter(os.path.join(self.RESULT_PATH, 'colinearity_result.xlsx'))
        corr_obj.corr_df.to_excel(writer, 'correlation_df', index=False)
        corr_obj.firstvif.to_excel(writer, 'INIT_VIF', index=False)
        corr_obj.lastvif.to_excel(writer, 'LAST_VIF', index=False)
        writer.save()


        self.summary_eda_table.loc[:, 'EDA筛查后保留'] = np.where(self.summary_eda_table.exclusion_reason.isnull(), 1, 0)
        self.summary_eda_table.loc[:, '综合算法排序后保留'] = np.where(self.summary_eda_table['指标英文'].isin(selected), 1, 0)

        colinearity_excluded_all = corr_obj.corr_exclude_vars + corr_obj.vif_dropvars
        colinearity_kept = [i for i in selected if i not in colinearity_excluded_all]
        self.summary_eda_table.loc[:, '共线性筛查后保留'] = np.where(self.summary_eda_table['指标英文'].isin(colinearity_kept), 1, 0)

        if len(colinearity_kept) > self.KEPT_LIMIT:
            logging.info("第二步进行变量选择: 变量筛选。共线性筛查后保留变量数量过多，进行stepwise进步筛选。")

            logger = logging.getLogger(__name__)
            # default是采取likelihood ratio test, 并且设置的alpha是0.05
            stepwise_result = fs.FeatureSelection().stepwise(self.X_transformed_train[colinearity_kept], self.y_train, logger, start_from=[])
            stepwise_result.to_excel(os.path.join(self.RESULT_PATH, 'stepwise_result.xlsx'), index=False)
            stepwise_kept = stepwise_result.loc[stepwise_result.final_selected==1, 'var_code'].unique()
            self.summary_eda_table.loc[:, '自动10分箱数据STEPWISE保留'] = np.where(self.summary_eda_table['指标英文'].isin(stepwise_kept), 1, 0)


        cols_filter = {}
        cols_filter['eda_exclude_cols'] = list(self.eda_exclude_cols)
        cols_filter['alg_filter_selected'] = list(selected)
        cols_filter['corr_exclude_cols'] = list(colinearity_excluded_all)
        if len(colinearity_kept) > self.KEPT_LIMIT:
            cols_filter['ranking_kept'] = list(stepwise_kept)
        else:
            cols_filter['ranking_kept'] = list(colinearity_kept)

        cols_filter['xgboost_candidate_pool'] = list(colinearity_kept)

        logging.info("第二步进行变量选择: 变量筛选。保留变量数量：%s" % len(cols_filter['ranking_kept']))
        save_data_to_json(cols_filter, self.RESULT_PATH, 'variable_filter.json')
        self.summary_eda_table.to_excel(os.path.join(self.RESULT_PATH, '%s_variables_summary.xlsx' % self.SAVE_LABEL), index=False)

        logging.info("第二步进行变量选择: 变量筛选。根据各阈值从综合排序中筛选留下的变量的共线性结果存储文件：%s" % os.path.join(self.RESULT_PATH, 'colinearity_result.xlsx'))
        logging.info("第二步进行变量选择: 变量筛选。不同步骤变量筛选过程保留、剔除变量列表文件存储于：%s" % os.path.join(self.RESULT_PATH, 'variable_filter.json'))
        logging.info("第二步进行变量选择: 变量筛选。变量筛选原因更新文件：%s" % os.path.join(self.RESULT_PATH, '%s_variables_summary.xlsx' % self.SAVE_LABEL))



class StepThreeCoarseCheck(object):
    """
    第三步检验粗分箱稳定性
    """
    def __init__(self, params):
        """
        Args:
        params (dict): dict 包含以下
        - all_x_y (pd.DataFrame()): 建模整理好的数据，缺失值已经处理好，通常包括每单申请的一些基本信息
            （如product_name, gamegroup, sample_set等），Y和X数据。另外，index必须为区分申请的unique id
            也必须包含columns:'sample_set', 用于区分train, test
        - var_dict (pd.DataFrame()): 标准变量字典表，包含以下这些列：数据源，数据类型，指标类型，
        指标英文，指标中文。
        - x_cols (list): X变量名list
        - y_col (str): column name of y

        - TRAIN_SET_NAME (str): all_x_y中'sample_set'列取值为train set的具体取值
        - TEST_SET_NAME (str): all_x_y中'sample_set'列取值为test set的具体取值, 如果为空则认为没有test set的需要
        - DATA_PATH (str): 数据存储路径
        - RESULT_PATH (str): 结果存储路径
        - TIME_COL (str): the column name indicating "time". e.g. 'apply_month', 'applicationdate'


        Returns:
        self.combined.to_excel(os.path.join(self.RESULT_PATH, '变量粗分箱按时间维度对比逾期率排序性.xlsx'), index=False)
        save_data_to_pickle({'X_cat_train': X_cat_train,
                            'X_transformed_train': X_transformed_train,
                            'X_cat_test': X_cat_test,
                            'X_transformed_test': X_transformed_test}, self.DATA_PATH, '建模粗分箱.pkl')

        Examples:
        step3 = StepThreeCoarseCheck(params)
        step3.compare_trend_over_time()
        trend_not_ok_exclude = []
        step3_obj.output(trend_not_ok_exclude)

        """
        self.all_x_y = params['all_x_y']
        if self.all_x_y.index.values[0] == 0:
            print("First index value = 0 discovered, please make sure that \
               the index is set to be the unique identifier, e.g. apply_id")
            raise
        if len(np.unique(self.all_x_y.index.values)) != len(self.all_x_y.index.values):
            print("Duplicated indices discovered, please make sure there \
                  are no duplicates")
            raise

        if 'sample_set' not in self.all_x_y.columns:
            print("'sample_set' column is not found in all_x_y, if no test \
                 set is required for the modeling, please set arg TEST=False")
            raise

        self.var_dict = params['var_dict']
        self.y_col = params['y_col']
        self.x_cols = params['x_cols']
        self.TRAIN_SET_NAME = params['TRAIN_SET_NAME']
        self.TEST_SET_NAME = params.get('TEST_SET_NAME', '')
        self.DATA_PATH = params['DATA_PATH']
        self.RESULT_PATH = params['RESULT_PATH']
        self.TIME_COL = params['TIME_COL']


        self.X_train = self.all_x_y.loc[self.all_x_y.sample_set==self.TRAIN_SET_NAME, self.x_cols].copy()
        self.y_train = self.all_x_y.loc[self.all_x_y.sample_set==self.TRAIN_SET_NAME, self.y_col].copy()

        if self.TEST_SET_NAME:
            self.X_test = self.all_x_y.loc[self.all_x_y.sample_set==self.TEST_SET_NAME, self.x_cols].copy()
            self.y_test = self.all_x_y.loc[self.all_x_y.sample_set==self.TEST_SET_NAME, self.y_col].copy()

        self.cols_filter = load_data_from_json(self.RESULT_PATH, 'variable_filter.json')
        self.logistic_candidate_pool = self.cols_filter['ranking_kept']



    def compare_trend_over_time(self):
        def _clean(data, col, result_col, rebin_spec):
            data.columns = [result_col+i for i in data.columns.values]
            data.index = pd.Series(data.index).astype(str)
            key_order = mt.BinWoe().order_bin_output(pd.DataFrame({'bin': data.index}), 'bin').bin.tolist()
            data = data.loc[key_order]
            data = data.reset_index().rename(columns={col:'分箱'})
            data.loc[:, '指标英文'] = col

            data.loc[:, '分箱对应原始分类'] = None
            if self.var_dict.loc[self.var_dict['指标英文']==col, '数据类型'].iloc[0]=='varchar' and col in rebin_spec:
                col_spec = deepcopy(rebin_spec[col])
                for new_label, original_x in list(col_spec.items()):
                    col_spec[str(new_label)] = ', '.join([str(i) for i in original_x])

                data.loc[:, '分箱对应原始分类'] = data['分箱'].astype('str')
                data.loc[:, '分箱对应原始分类'] = data['分箱对应原始分类'].replace(col_spec)

            return data

        def _clean2(data_list):
            data = pd.concat(data_list)
            data = data.merge(self.var_dict[['数据源', '指标英文', '指标中文']], on='指标英文')

            header_columns = ['数据源', '指标英文', '指标中文', '分箱', '分箱对应原始分类']
            other_columns = [i for i in data.columns if i not in header_columns]
            columns_order = header_columns + other_columns
            return data[columns_order]


        logging.info("第三步检验粗分箱稳定性: 按时间维度检查排序性稳定。所用数据集：TRAIN")
        try:
            coarse_classing_rebin_spec = load_data_from_pickle(self.DATA_PATH, 'coarse_classing_rebin_spec.pkl')
        except:
            coarse_classing_rebin_spec = load_data_from_pickle(self.DATA_PATH, '建模细分箱10结果.pkl')['rebin_spec']

        rebin_spec = {k:v for k,v in coarse_classing_rebin_spec.items() if k in self.logistic_candidate_pool}


        X_cat_train = mt.BinWoe().convert_to_category(self.X_train[self.logistic_candidate_pool],
                                                      self.var_dict, rebin_spec)

        merged_with_time = X_cat_train.merge(self.all_x_y[[self.TIME_COL, self.y_col]], left_index=True, right_index=True)

        eventrate_list = []
        ct_list = []
        pct_list = []
        for col in self.logistic_candidate_pool:
            logging.log(18, col + " starts event rate by time calculation")
            eventrate_time_pivot = merged_with_time.pivot_table(index=col, columns=self.TIME_COL, values=self.y_col)
            eventrate_time_pivot = _clean(eventrate_time_pivot, col, '逾期率 ', rebin_spec)
            eventrate_list.append(eventrate_time_pivot)

            ct_time_pivot = merged_with_time.pivot_table(index=col, columns=self.TIME_COL, values=self.y_col, aggfunc='count')
            pct_time_pivot = ct_time_pivot.fillna(0)/ct_time_pivot.sum()
            ct_time_pivot = _clean(ct_time_pivot, col, '样本量 ', rebin_spec)
            pct_time_pivot = _clean(pct_time_pivot, col, '分布占比 ', rebin_spec)

            ct_list.append(ct_time_pivot)
            pct_list.append(pct_time_pivot)


        writer = pd.ExcelWriter(os.path.join(self.RESULT_PATH, '变量粗分箱按时间维度对比逾期率排序性.xlsx'))
        _clean2(eventrate_list).to_excel(writer, '逾期率', index=False)
        _clean2(ct_list).to_excel(writer, '样本量', index=False)
        _clean2(pct_list).to_excel(writer, '分布占比', index=False)
        writer.save()

        logging.info("第三步检验粗分箱稳定性: 按时间维度检查排序性稳定。结果文件存储于：%s" % os.path.join(self.RESULT_PATH, '变量粗分箱按时间维度对比逾期率排序性.xlsx'))



    def output(self, trend_not_ok_exclude=[]):
        logging.info("第三步检验粗分箱稳定性: 输出最终粗分箱结果。所用数据集：TRAIN & TEST")

        try:
            coarse_classing_rebin_spec = load_data_from_pickle(self.DATA_PATH, 'coarse_classing_rebin_spec.pkl')
        except:
            coarse_classing_rebin_spec = load_data_from_pickle(self.DATA_PATH, '建模细分箱10结果.pkl')['rebin_spec']

        coarse_classing_rebin_spec = {k:v for k,v in coarse_classing_rebin_spec.items() if k in self.logistic_candidate_pool and k not in trend_not_ok_exclude}

        self.cols_filter['coarsebin_kept_after_ranking'] = [i for i in self.logistic_candidate_pool if i not in trend_not_ok_exclude]
        save_data_to_json(self.cols_filter, self.RESULT_PATH, 'variable_filter.json')

        X_cat_train = mt.BinWoe().convert_to_category(self.X_train[self.cols_filter['coarsebin_kept_after_ranking']],
                                                      self.var_dict, coarse_classing_rebin_spec)
        woe_iv_df_coarse = mt.BinWoe().calculate_woe_all(X_cat_train, self.y_train,
                                                self.var_dict, coarse_classing_rebin_spec)
        woe_iv_df_coarse.to_excel(os.path.join(self.RESULT_PATH, 'woe_iv_df_coarse.xlsx'), index=False)

        X_transformed_train = mt.BinWoe().transform_x_all(X_cat_train, woe_iv_df_coarse)

        to_save_data = {'X_cat_train': X_cat_train,
                        'X_transformed_train': X_transformed_train}

        if self.TEST_SET_NAME:
            X_cat_test = mt.BinWoe().convert_to_category(self.X_test[self.cols_filter['coarsebin_kept_after_ranking']],
                                                    self.var_dict, coarse_classing_rebin_spec)
            X_transformed_test = mt.BinWoe().transform_x_all(X_cat_test, woe_iv_df_coarse)
            to_save_data['X_cat_test'] = X_cat_test
            to_save_data['X_transformed_test'] = X_transformed_test

        save_data_to_pickle(to_save_data, self.DATA_PATH, '建模粗分箱.pkl')

        logging.info("第三步检验粗分箱稳定性: 输出最终粗分箱结果。变量粗分箱分箱边界文件：%s" % os.path.join(self.DATA_PATH, 'coarse_classing_rebin_spec.pkl'))
        logging.info("第三步检验粗分箱稳定性: 输出最终粗分箱结果。TRAIN&TEST数据粗分箱分箱数据、WOE转换数据分别存储于：%s" % os.path.join(self.DATA_PATH, '建模粗分箱.pkl'))
        logging.info("第三步检验粗分箱稳定性: 输出最终粗分箱结果。更新不同步骤变量筛选过程保留、剔除变量列表文件存储于：%s" % os.path.join(self.RESULT_PATH, 'variable_filter.json'))
        logging.info("第三步检验粗分箱稳定性: 输出最终粗分箱结果。变量粗分箱WOE-IV文件：%s" % os.path.join(self.RESULT_PATH, 'woe_iv_df_coarse.xlsx'))




class StepFourTrain(object):
    """
    第四步粗分箱后进行建模, 先在TRAIN train上stepwise筛选显著且coef不是负的变量
    如果TEST set的样本量也比较大 (>5000)，那么在如此大的样本上我们认为模型在这个样本上
    如果表现稳定的话应该coef不为负且也是显著的，所以如果有字段不符合要求将会被删除
    最终模型存储model obj，系数，KS，AUC图等结果
    """
    def __init__(self, params):
        """
        Args:
        params (dict): dict 包含以下
        - all_x_y (pd.DataFrame()): 建模整理好的数据，缺失值已经处理好，通常包括每单申请的一些基本信息
            （如product_name, gamegroup, sample_set等），Y和X数据。另外，index必须为区分申请的unique id
            也必须包含columns:'sample_set', 用于区分train, test
        - var_dict (pd.DataFrame()): 标准变量字典表，包含以下这些列：数据源，数据类型，指标类型，
        指标英文，指标中文。
        - x_cols (list): X变量名list
        - y_col (str): column name of y
        - xgbparams_selection_method (list): ['XGBExp', 'XGBGrid', 'XGBRandom'],如果有xgboost模型的时候
        - xgb_params_range (dict): xgboost gridsearch/randomsearch的搜索范围区间
            e.g. xgb_params_range = {
                    'learning_rate': [0.05,0.1,0.2],
                    'max_depth': [2, 3],
                    'gamma': [0, 0.1],
                    'min_child_weight':[1,3,10,20],
                    # 'subsample': np.linspace(0.4, 0.7, 3),
                    # 'colsample_bytree': np.linspace(0.4, 0.7, 3),
                }
        - xgb_param_experience (dict): xgboost自定义经验参数
            e.g. xgb_param_experience = {
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

        - TRAIN_SET_NAME (str): all_x_y中'sample_set'列取值为train set的具体取值
        - TEST_SET_NAME (str): all_x_y中'sample_set'列取值为test set的具体取值, 如果为空则认为没有test set的需要
        - OOT_SET_NAME (str): all_x_y中'sample_set'列取值为OOT set的具体取值, 如果为空则认为没有oot set
        - NFOLD (int): default=5, validate表现的时候的CV的折数
        - DATA_PATH (str): 数据存储路径
        - RESULT_PATH (str): 结果存储路径
        - TIME_COL (str): the column name indicating "time". e.g. 'apply_month', 'applicationdate'


        Examples:
        step4_obj = StepFourTrain(params)
        # 任选以下几种
        step4_obj.run(model_names=['logistic'])
        step4_obj.run(model_names=['xgboost'])
        step4_obj.run(model_names=['logistic', 'xgboost'])
        """
        self.all_x_y = params['all_x_y']
        if self.all_x_y.index.values[0] == 0:
            print("First index value = 0 discovered, please make sure that \
               the index is set to be the unique identifier, e.g. apply_id")
            raise
        if len(np.unique(self.all_x_y.index.values)) != len(self.all_x_y.index.values):
            print("Duplicated indices discovered, please make sure there \
                  are no duplicates")
            raise

        if 'sample_set' not in self.all_x_y.columns:
            print("'sample_set' column is not found in all_x_y, if no test \
                 set is required for the modeling, please set arg TEST=False")
            raise

        self.var_dict = params['var_dict']
        self.x_cols = params['x_cols']
        self.y_col = params['y_col']
        self.xgbparams_selection_method = params.get('xgbparams_selection_method', ['XGBExp'])
        self.xgb_params_range = params.get('xgb_params_range', None)
        self.xgb_param_experience = params.get('xgb_param_experience', None)

        self.TRAIN_SET_NAME = params['TRAIN_SET_NAME']
        self.TEST_SET_NAME = params.get('TEST_SET_NAME', '')
        self.OOT_SET_NAME = params['OOT_SET_NAME']
        self.DATA_PATH = params['DATA_PATH']
        self.RESULT_PATH = params['RESULT_PATH']
        self.TIME_COL = params['TIME_COL']
        self.NFOLD = params.get('NFOLD', 5)

        self.X_train = self.all_x_y.loc[self.all_x_y.sample_set==self.TRAIN_SET_NAME, self.x_cols].copy()
        self.y_train = self.all_x_y.loc[self.all_x_y.sample_set==self.TRAIN_SET_NAME, self.y_col].copy()

        if self.TEST_SET_NAME:
            self.X_test = self.all_x_y.loc[self.all_x_y.sample_set==self.TEST_SET_NAME, self.x_cols].copy()
            self.y_test = self.all_x_y.loc[self.all_x_y.sample_set==self.TEST_SET_NAME, self.y_col].copy()

        self.cols_filter = load_data_from_json(self.RESULT_PATH, 'variable_filter.json')
        if 'coarsebin_kept_after_ranking' in self.cols_filter:
            self.logistic_candidate_pool = self.cols_filter['coarsebin_kept_after_ranking']
        else:
            self.logistic_candidate_pool = self.cols_filter['ranking_kept']

        self.xgboost_candidate_pool = self.cols_filter['xgboost_candidate_pool']

        self.coarse_data_dict = load_data_from_pickle(self.DATA_PATH, '建模粗分箱.pkl')

        if sum(self.X_train.index == self.coarse_data_dict['X_transformed_train'].index) != len(self.X_train):
            print("X_train, X_train_transformed index not matched")
            raise

        if sum(self.y_train.index == self.coarse_data_dict['X_transformed_train'].index) != len(self.y_train):
            print("y_train, X_train_transformed index not matched")
            raise

        if self.TEST_SET_NAME:
            if sum(self.X_test.index == self.coarse_data_dict['X_transformed_test'].index) != len(self.X_test):
                print("X_test, X_transformed_test index not matched")
                raise

            if sum(self.y_test.index == self.coarse_data_dict['X_transformed_test'].index) != len(self.y_test):
                print("y_test, X_transformed_test index not matched")
                raise




    def run(self, model_names=['logistic']):
        """
        Args:
        model_name (list): default=['logistic']

        """
        for model_name in model_names:
            if model_name == 'logistic':
                logging.info("第四步：建模。逻辑回归")
                self.stepwise()
                if self.TEST_SET_NAME and len(self.y_test) > 5000:
                    self.verify_test()

                self.logit_model_wrap('AutoDefaultLogistic')

            if model_name == 'xgboost':
                logging.info("第四步：建模。XGBoost")
                self.process_xgbdata('AutoDefault')
                self.xgb_wrap('AutoDefault')



    def stepwise(self):
        logging.info("第四步：建模。逻辑回归：粗分箱变量进行STEPWISE变量选择")

        logger = logging.getLogger(__name__)
        stepwise_result = fs.FeatureSelection().stepwise(self.coarse_data_dict['X_transformed_train'][self.logistic_candidate_pool],
                                        self.y_train, logger, start_from=[])

        self.in_model = stepwise_result.loc[stepwise_result.final_selected==1, 'var_code'].unique()
        self.cols_filter['stepwise_selected'] = list(self.in_model)
        save_data_to_json(self.cols_filter, self.RESULT_PATH, 'variable_filter.json')

        logging.info("第四步：建模。逻辑回归：粗分箱变量进行STEPWISE变量选择, 选中变量列表更新于：%s" % os.path.join(self.RESULT_PATH, 'variable_filter.json'))



    def verify_test(self):
        """
        如果TEST set的样本量也比较大，那么在如此大的样本上我们认为模型在这个样本上如果表现稳定的
        话应该coef不为负且也是显著的，所以如果有字段不符合要求将会被删除
        """
        logging.info("第四步：建模。逻辑回归：当TEST数据集样本量较大5000将会通过在TEST样本上STEPWISE确保TRAIN样本上选中的变量在TEST上同样显著")

        logger = logging.getLogger(__name__)
        stepwise_result = fs.FeatureSelection().stepwise(self.coarse_data_dict['X_transformed_test'][self.in_model],
                                        self.y_test, logger, start_from=[])

        self.in_model = stepwise_result.loc[stepwise_result.final_selected==1, 'var_code'].unique()
        self.cols_filter['stepwise_selected'] = list(self.in_model)
        save_data_to_json(self.cols_filter, self.RESULT_PATH, 'variable_filter.json')

        logging.info("第四步：建模。逻辑回归：粗分箱变量进行TEST集验证STEPWISE变量选择, 选中变量列表更新于：%s" % os.path.join(self.RESULT_PATH, 'variable_filter.json'))


    def logit_model_wrap(self, model_label, woe_file_name='woe_iv_df_coarse.xlsx',
                            rebin_spec_name='coarse_classing_rebin_spec.pkl',
                            in_model=[]):
        """
        此函数可以单独摘出来，在后续参照OOT表现和业务调整时使用
        woe_file_name (str): default='woe_iv_df_coarse.xlsx'。如果有其他分箱边界产生
            的分箱文件可以传入相应的名字
        rebin_spec_name (str): 变量粗分箱（或者评分卡建模用的分箱）边界spec文件名。用于
            保证评分卡生成过程中变量分箱枚举值完整
            default='coarse_classing_rebin_spec.pkl'。如果有其他分箱边界可传入相应的名字
        in_model (list): default=[], 如果与从TRAIN、TEST stepwise
                            (p值，coef>0)后选中的不同，可单独传入
        """
        # 不直接用self.in_model是为了保证这个函数单独使用的时候，并没有执行self.stepwise
        # 和self.verify_test故而没有self.in_model
        if len(in_model) == 0:
            in_model = self.cols_filter['stepwise_selected']

        if self.TEST_SET_NAME:
            modeling_obj = ml.LogisticModel(var_dict=self.var_dict, y_train=self.y_train,
                                    y_test=self.y_test)
            model_result = modeling_obj.fit_model(model_label,
                            self.coarse_data_dict['X_transformed_train'][in_model],
                            self.coarse_data_dict['X_transformed_test'][in_model], \
                            in_model, nfold=self.NFOLD)
            modeling_obj.plot_for_the_model(self.RESULT_PATH, model_result, with_test=True)
        else:
            modeling_obj = ml.LogisticModel(var_dict=self.var_dict, y_train=self.y_train,
                                    with_test=False)
            model_result = modeling_obj.fit_model(model_label,
                            self.coarse_data_dict['X_transformed_train'][in_model],
                            self.coarse_data_dict['X_transformed_train'][in_model],
                            in_model, nfold=self.NFOLD)

            modeling_obj.plot_for_the_model(self.RESULT_PATH, model_result, with_test=False)

        modeling_obj.model_save(self.DATA_PATH, self.RESULT_PATH)
        vp.get_score_card(model_label, self.DATA_PATH, self.RESULT_PATH,
                          woe_file_name, rebin_spec_name, self.var_dict)

        pl.plot_colinearity(self.coarse_data_dict['X_transformed_train'][in_model].corr(),
                            self.RESULT_PATH, 'ModelSelected')

        logging.info("第四步：建模。逻辑回归。模型obj，probility等存储于 %s" % os.path.join(self.DATA_PATH, '%s模型结果.pkl' % model_label))
        logging.info("第四步：建模。逻辑回归。TRAIN集模型系数存储于 %s" % os.path.join(self.RESULT_PATH, '%s_train_coef.xlsx' % model_label))
        logging.info("第四步：建模。逻辑回归。TEST集模型系数存储于 %s" % os.path.join(self.RESULT_PATH, '%s_test_coef.xlsx' % model_label))
        logging.info("第四步：建模。逻辑回归。模型评分卡存储于 %s" % os.path.join(self.RESULT_PATH, '%s_score_card.xlsx' % model_label))
        logging.info("第四步：建模。逻辑回归。各样本KS、AUC图存储于 %s" % os.path.join(self.RESULT_PATH, 'figure'))






    def process_xgbdata(self, model_label_version):
        logging.info("第四步：建模。XGBoost：变量拓展衍生：连续变量原始值、连续变量分箱值、分类变量dummy转化")

        if len(self.X_train) > 15000 and sum(self.y_train) > 1000:
            X_train_xgboost, auto_rebin_spec, bin_to_label, dummy_var_name_map = \
                    mt.BinWoe().xgboost_data_derive(self.X_train[self.logistic_candidate_pool], self.y_train,
                        self.var_dict, num_max_bins=20)
        else:
            X_train_xgboost, auto_rebin_spec, bin_to_label, dummy_var_name_map = \
                    mt.BinWoe().xgboost_data_derive(self.X_train[self.xgboost_candidate_pool], self.y_train,
                        self.var_dict, num_max_bins=10)

        if self.TEST_SET_NAME:
            X_test_xgboost = mt.BinWoe().apply_xgboost_data_derive(self.X_test[self.xgboost_candidate_pool],
                                self.var_dict, auto_rebin_spec, bin_to_label)
            save_data_to_pickle({
            'X_train_xgboost': X_train_xgboost,
            'X_test_xgboost': X_test_xgboost,
            'auto_rebin_spec': auto_rebin_spec,
            'bin_to_label': bin_to_label,
            'dummy_var_name_map': dummy_var_name_map}, self.DATA_PATH, '%s_XGBoost输出数据和分箱明细.pkl' % model_label_version)

        else:
            save_data_to_pickle({
            'X_train_xgboost': X_train_xgboost,
            'auto_rebin_spec': auto_rebin_spec,
            'bin_to_label': bin_to_label,
            'dummy_var_name_map': dummy_var_name_map}, self.DATA_PATH, '%s_XGBoost输出数据和分箱明细.pkl' % model_label_version)

        logging.info("第四步：建模。XGBoost：TRAIN，TEST集准备好的XGB数据，分箱边界，以及分类变量转换成数值label时的对照表存储于%s"
                    % os.path.join(self.DATA_PATH, '%s_XGBoost输出数据和分箱明细.pkl' % model_label_version))


    def xgb_wrap(self, model_label_version):
        if self.TEST_SET_NAME:
            xgbmodel_obj = ml.XGBModel(self.var_dict, self.DATA_PATH, self.RESULT_PATH,
                        model_label_version, self.y_train,
                        y_test=self.y_test,
                        params_selection_method=self.xgbparams_selection_method,
                        nfold=self.NFOLD)
            xgbmodel_obj.run(self.xgb_params_range, self.xgb_param_experience)

            plot_obj = ml.LogisticModel(var_dict=self.var_dict, y_train=self.y_train,
                                    y_test=self.y_test)

            for model_label in self.xgbparams_selection_method:
                model_result = load_data_from_pickle(self.DATA_PATH,
                                '%s模型结果.pkl' % (model_label_version+model_label))
                plot_obj.plot_for_the_model(self.RESULT_PATH, model_result, with_test=True)


        else:
            xgbmodel_obj = ml.XGBModel(self.var_dict, self.DATA_PATH, self.RESULT_PATH,
                        model_label_version, self.y_train,
                        params_selection_method=self.xgbparams_selection_method,
                        nfold=self.NFOLD)
            xgbmodel_obj.run(self.xgb_params_range, self.xgb_param_experience)

            plot_obj = ml.LogisticModel(var_dict=self.var_dict, y_train=self.y_train,
                                    with_test=False)

            for model_label in self.xgbparams_selection_method:
                model_result = load_data_from_pickle(self.DATA_PATH,
                                '%s模型结果.pkl' % (model_label_version+model_label))
                plot_obj.plot_for_the_model(self.RESULT_PATH, model_result, with_test=False)

        logging.info("第四步：建模。XGBoost模型AUC、KS图存储于文件夹 %s 中" % os.path.join(self.RESULT_PATH, 'figure'))




class StepFiveVerify(object):
    """
    对比表现
    """
    def __init__(self, params):
        """
        Args:
        params (dict): dict 包含以下
        - data_cat_dict (dict): key为sample_set的名称。value为分箱好的数据，index为
            区分每条数据的unique identifier。用于逻辑回归计算分数
        - xgboost_data_dict (dict): key为sample_set的名称。value为xgboost要用的数据，
            index为区分每条数据的unique identifier。
        - var_dict (pd.DataFrame()): 标准变量字典表，包含以下这些列：数据源，数据类型，指标类型，
        指标英文，指标中文。
        - all_Y (pd.DataFrame): 所有数据的(TRAIN, TEST, OOT etc)的Y，不同表现期的Y
            index为区分每条数据的unique identifier
        - y_cols (list): all_Y中需要用于对比表现的Y的column name
        - union_var_list (list): 所有需要对比的模型所选中的变量，只限逻辑回归模型
        - model_labels (list): 需要对比的模型model_label名称
        - backscore_has_perf (list): list of applyid (unique identifier)用于区分
            backscore样本中有表现的样本。default=[]。当传入的数据用backscore set时，此为
            必选
        - base_model_score (dict): 用于对比的老模型的分数，key为sample_set的名称。
            value为分数，index为区分每条数据的unique identifier。存储形式为pd.Series。
            如果需要与老模型，或者有老模型做为对比时，此为必选
        - BASE_MODEL_LABEL (str): 用于对比的老模型或者benchmark模型。如果需要与老模型，
            或者有老模型做为对比时，此为必选
        - coarse_classing_rebin_spec (dict): 粗分箱spec
        - liftcurve_name_map (dict): 画lift curve的时候原始model_label转换成图中的
            显示名称，因为画图会将中文显示成方格，所以转换的值需全为英文。取名时请注意规范一致
            性和可读性和理解性，因为这个是会放到最终报告中的。如果有用于对比表现的老模型，也
            需要包含。key值为建模时各模型对应的model_label, value值为规范刻度和解释性较好
            的全英文正式模型名称

        - TRAIN_SET_NAME (str): 'sample_set'列取值为train set的具体取值
        - TEST_SET_NAME (str): 'sample_set'列取值为test set的具体取值
        - OOT_SET_NAME (str): all_x_y中'sample_set'列取值为OOT set的具体取值, 如果为空则认为没有oot set
        - BACKSCORE_SET_NAME (str): 'sample_set'列取值为backscore set的具体取值
        - DATA_PATH (str): 数据存储路径
        - RESULT_PATH (str): 结果存储路径
        - Y_BUILD_NAME (str): 建模所用的Y的列明


        Examples:
        step5_obj = StepFiveVerify(params)
        step5_obj.run()

        """
        self.data_cat_dict = params['data_cat_dict']
        self.xgboost_data_dict = params.get('xgboost_data_dict', {})
        self.all_Y = params['all_Y']
        self.var_dict = params['var_dict']
        self.y_cols = params['y_cols']
        self.backscore_has_perf = params.get('backscore_has_perf', [])
        self.base_model_score = params.get('base_model_score', {})
        self.union_var_list = params.get('union_var_list', [])
        self.model_labels = params.get('model_labels', [])
        self.coarse_classing_rebin_spec = params['coarse_classing_rebin_spec']
        self.liftcurve_name_map = params['liftcurve_name_map']



        self.TRAIN_SET_NAME = params['TRAIN_SET_NAME']
        self.TEST_SET_NAME = params.get('TEST_SET_NAME', '')
        self.OOT_SET_NAME = params['OOT_SET_NAME']
        self.BACKSCORE_SET_NAME = params['BACKSCORE_SET_NAME']
        self.DATA_PATH = params['DATA_PATH']
        self.RESULT_PATH = params['RESULT_PATH']
        self.BASE_MODEL_LABEL = params.get('BASE_MODEL_LABEL', '')
        self.Y_BUILD_NAME = params['Y_BUILD_NAME']


    def run(self):
        self.variable_stability()
        self.variable_ranking_comparison()
        self.generate_score()
        self.generate_ksauc_comparison_table()
        self.generate_decile()
        self.lift_curve()
        self.swap_analysis()
        self.score_stability()
        if len(self.backscore_has_perf) > 0:
            self.backscore_approval_rate_analysis()

    def variable_stability(self):
        logging.info("第五步模型验证：变量稳定性")
        perf = mt.Performance()
        var_psi_list = []
        compare_set_names = [i for i in list(self.data_cat_dict.keys()) if i != self.TRAIN_SET_NAME]
        for label in compare_set_names:
            var_psi_train_vs = perf.variable_psi(self.data_cat_dict[self.TRAIN_SET_NAME][self.union_var_list], \
                                                self.data_cat_dict[label][self.union_var_list], self.var_dict)
            columns_order = var_psi_train_vs.columns
            var_psi_train_vs.loc[:, 'compare_set'] = label
            var_psi_list.append(var_psi_train_vs)

        columns_order = ['compare_set'] + columns_order
        var_psi = pd.concat(var_psi_list)

        var_psi.to_excel(os.path.join(self.RESULT_PATH, 'all_variable_psi.xlsx'), index=False)
        logging.info("第五步模型验证：变量稳定性PSI存储于 %s" % os.path.join(self.RESULT_PATH, 'all_variable_psi.xlsx'))



    def variable_ranking_comparison(self):
        logging.info("第五步模型验证：变量排序性稳定性。对比TRAIN vs OOT样本于不同Y定义下的排序性")
        for y_col in self.y_cols:
            base_cat = self.data_cat_dict[self.TRAIN_SET_NAME]\
                            .merge(self.all_Y[[y_col]], left_index=True, right_index=True)
            compare_cat = self.data_cat_dict[self.OOT_SET_NAME]\
                            .merge(self.all_Y[[y_col]], left_index=True, right_index=True)

            base_cat = base_cat.loc[base_cat[y_col].notnull()]
            compare_cat = compare_cat.loc[compare_cat[y_col].notnull()]


            if len(base_cat) > 0 and len(compare_cat) > 0:
                perf_badrate = pl.variable_badrate_compare(base_cat[self.union_var_list],
                                                        base_cat[y_col],
                                                        compare_cat[self.union_var_list],
                                                        compare_cat[y_col],
                                                        self.TRAIN_SET_NAME,
                                                        self.OOT_SET_NAME,
                                                        self.var_dict,
                                                        self.coarse_classing_rebin_spec,
                                                        self.RESULT_PATH, plot=False)

                to_save_path_name = os.path.join(self.RESULT_PATH, '变量%s逾期率对比%sVs%s.xlsx' % (y_col, self.TRAIN_SET_NAME, self.OOT_SET_NAME))
                perf_badrate.to_excel(to_save_path_name, index=False)
                logging.info("第五步模型验证：变量排序性稳定性。结果文件存储于%s" % to_save_path_name)


    def generate_score(self):
        logging.info("第五步模型验证：计算各模型在不同数据集上的分数")
        for model_label in self.model_labels:
            score_dict = {}
            if 'XGB' not in model_label:
                score_card = pd.read_excel(os.path.join(self.RESULT_PATH, '%s_score_card.xlsx' % model_label))

                for set_name, dd in self.data_cat_dict.items():
                    varscore, score = mt.Performance().calculate_score_by_scrd(dd, score_card)
                    score_dict[set_name] = {'varscore': varscore, 'score': score}

                save_data_to_pickle(score_dict, self.DATA_PATH, '模型%s分数.pkl' % model_label)
                logging.info("第五步模型验证：分数存储于 %s" % os.path.join(self.DATA_PATH, '模型%s分数.pkl' % model_label))

            else:
                model_result = load_data_from_pickle(self.DATA_PATH, '%s模型结果.pkl' % model_label)
                score_dict = {}
                for set_name, dd in self.xgboost_data_dict.items():
                    dd_dmatrix = xgb.DMatrix(dd)
                    p = pd.Series(model_result['model_final'].predict(dd_dmatrix), index=dd.index)
                    score = p.apply(mt.Performance().p_to_score)
                    score_dict[set_name] = {'score': score, 'prob': p}

                save_data_to_pickle(score_dict, self.DATA_PATH, '模型%s分数.pkl' % model_label)
                logging.info("第五步模型验证：分数存储于 %s" % os.path.join(self.DATA_PATH, '模型%s分数.pkl' % model_label))



    def generate_ksauc_comparison_table(self):
        logging.info("第五步模型验证：计算统计各模型各样本上的KS & AUC")

        r_list = []
        for model_label in self.model_labels:
            score_dict = load_data_from_pickle(self.DATA_PATH, '模型%s分数.pkl' % model_label)
            model_result = load_data_from_pickle(self.DATA_PATH, '%s模型结果.pkl' % model_label)
            for y_col in self.y_cols:
                perf_result = vp.summarize_perf_metrics(score_dict, self.all_Y, y_col)

                cv_prob_with_y = model_result['all_p_cv'].to_frame('prob')\
                                   .merge(self.all_Y, left_index=True, right_index=True)
                cv_prob_with_y.loc[:, 'score'] = cv_prob_with_y.prob.apply(mt.Performance().p_to_score)
                set_name = '%sFoldCV' % model_result['nfold']
                auc = mt.Performance().calculate_auc(cv_prob_with_y[y_col], cv_prob_with_y.prob)
                ks = mt.Performance().calculate_ks_by_score(cv_prob_with_y[y_col], cv_prob_with_y.score)

                logging.log(18, model_label)
                logging.log(18, y_col)
                perf_result.loc[:, set_name] = None
                perf_result.loc[perf_result.metrics=='AUC', set_name] = auc
                perf_result.loc[perf_result.metrics=='KS', set_name] = ks

                perf_result.loc[:, 'Y_definition'] = y_col
                perf_result.loc[:, 'model_label'] = model_label
                r_list.append(perf_result)

        all_perf_summary = pd.concat(r_list)
        all_perf_summary['model_label'] = all_perf_summary['model_label'].replace(self.liftcurve_name_map)

        columns_order = ['model_label', 'Y_definition', 'metrics', self.TRAIN_SET_NAME,
                        self.TEST_SET_NAME, self.OOT_SET_NAME, set_name]
        other_columns = [i for i in all_perf_summary.columns if i not in columns_order]
        columns_order = columns_order + other_columns
        columns_order = [i for i in columns_order if i in all_perf_summary.columns]

        all_perf_summary[columns_order].sort_values(['Y_definition', 'metrics', 'model_label'])\
                .to_excel(os.path.join(self.RESULT_PATH, 'all_ksauc_perf_summary.xlsx'), index=False)
        logging.info("第五步模型验证：各模型各样本上的KS & AUC结果存储于 %s" % os.path.join(self.RESULT_PATH, 'all_ksauc_perf_summary.xlsx'))


    def generate_decile(self):
        logging.info("第五步模型验证：计算各模型各样本decile")
        ks_decile_list = []
        for model_label in self.model_labels:
            score_dict = load_data_from_pickle(self.DATA_PATH, '模型%s分数.pkl' % model_label)
            logging.log(18, "generate_decile: model %s" % model_label)

            for y_col in self.y_cols:
                logging.log(18, y_col)
                # 取 train的decile bounds用于应该用到所有其他样本上
                train_score = score_dict[self.TRAIN_SET_NAME]
                if 'XGB' in model_label:
                    all_score = train_score['score']\
                                  .to_frame('score')\
                                  .merge(self.all_Y[[y_col]], left_index=True, right_index=True)
                    all_score = all_score.loc[all_score[y_col].notnull()]
                    train_decile = mt.Performance().calculate_ks_by_decile(all_score.score, all_score[y_col], 'decile', 10)
                    decile_train_bounds = mt.BinWoe().obtain_boundaries(train_decile['分箱'])['cut_boundaries']

                else:
                    _, decile_train_bounds = vp.get_decile(train_score['score'],
                                        train_score['varscore'], self.all_Y,
                                        y_col, self.TRAIN_SET_NAME, self.RESULT_PATH)

                # 计算decile
                for label, score_dd in score_dict.items():
                    logging.log(18, label)
                    if 'XGB' in model_label:
                        all_score = score_dd['score']\
                                      .to_frame('score')\
                                      .merge(self.all_Y[[y_col]], left_index=True, right_index=True)
                        all_score = all_score.loc[all_score[y_col].notnull()]
                        if len(all_score) > 10:
                            score_decile = mt.Performance().calculate_ks_by_decile(all_score.score, all_score[y_col],
                                                            'decile', manual_cut_bounds=decile_train_bounds)
                            score_decile['sample_set'] = label
                        else:
                            score_decile = pd.DataFrame()

                    else:
                        score_decile, _ = vp.get_decile(score_dd['score'],
                                                          score_dd['varscore'],
                                                          self.all_Y, y_col, label,
                                                          self.RESULT_PATH,
                                                          decile_train_bounds)

                    score_decile['Y_definition'] = y_col
                    score_decile['model_label'] = model_label
                    ks_decile_list.append(score_decile)


                # runbook
                if self.TEST_SET_NAME:
                    all_score = pd.concat([score_dict[self.TRAIN_SET_NAME]['score'], score_dict[self.TEST_SET_NAME]['score']])\
                                  .to_frame('score')\
                                  .merge(self.all_Y[[y_col]], left_index=True, right_index=True)
                    all_score = all_score.loc[all_score[y_col].notnull()]
                    runbook = mt.Performance().calculate_ks_by_decile(all_score.score, all_score[y_col], 'decile', 20)
                    point_bounds = mt.BinWoe().obtain_boundaries(runbook['分箱'])['cut_boundaries']
                    runbook = mt.Performance().calculate_ks_by_decile(all_score.score, all_score[y_col], 'decile', manual_cut_bounds=point_bounds)
                    decile_column_order = runbook.columns
                    runbook.loc[:, 'sample_set'] = 'RUNBOOK'
                    runbook.loc[:, 'Y_definition'] = y_col
                    runbook.loc[:, 'model_label'] = model_label
                    ks_decile_list.append(runbook)
                else:
                    all_score = score_dict[self.TRAIN_SET_NAME]['score']\
                                  .to_frame('score')\
                                  .merge(self.all_Y[[y_col]], left_index=True, right_index=True)
                    all_score = all_score.loc[all_score[y_col].notnull()]
                    runbook = mt.Performance().calculate_ks_by_decile(all_score.score, all_score[y_col], 'decile', 20)
                    point_bounds = mt.BinWoe().obtain_boundaries(runbook['分箱'])['cut_boundaries']
                    runbook = mt.Performance().calculate_ks_by_decile(all_score.score, all_score[y_col], 'decile', manual_cut_bounds=point_bounds)
                    decile_column_order = runbook.columns
                    runbook.loc[:, 'sample_set'] = 'RUNBOOK'
                    runbook.loc[:, 'Y_definition'] = y_col
                    runbook.loc[:, 'model_label'] = model_label
                    ks_decile_list.append(runbook)



        if len(self.base_model_score) > 0 and self.BASE_MODEL_LABEL:
            for sample_set, score in self.base_model_score.items():
                for y_col in self.y_cols:
                    score_with_y = score.to_frame('score').merge(self.all_Y[[y_col]],
                                                left_index=True, right_index=True)
                    score_with_y = score_with_y.loc[score_with_y[y_col].notnull()]
                    if len(score_with_y) > 10:
                        decile = mt.Performance().calculate_ks_by_decile(score_with_y.score,
                                                        score_with_y[y_col], 'decile', 10)
                        decile.loc[:, 'model_label'] = self.BASE_MODEL_LABEL
                        decile.loc[:, 'Y_definition'] = y_col
                        decile.loc[:, 'sample_set'] = sample_set
                        ks_decile_list.append(decile)

        decile_column_order = ['model_label', 'Y_definition', 'sample_set'] + list(decile_column_order)
        ks_decile_build = pd.concat(ks_decile_list)

        other_columns = sorted([i for i in ks_decile_build.columns if i not in decile_column_order])
        ks_decile_build = ks_decile_build[decile_column_order + other_columns]
        ks_decile_build['model_label'] = ks_decile_build['model_label'].replace(self.liftcurve_name_map)
        ks_decile_build.to_excel(os.path.join(self.RESULT_PATH, 'all_decile.xlsx'), index=False)
        logging.info("第五步模型验证：各模型各样本上Decile&Runbook结果存储于 %s" % os.path.join(self.RESULT_PATH, 'all_decile.xlsx'))



    def lift_curve(self):
        logging.info('第五步模型验证：lift curves对比各模型表现')
        all_decile_df = pd.read_excel(os.path.join(self.RESULT_PATH, 'all_decile.xlsx'))
        # 统一个sample set 对比不同模型
        for sample_set in all_decile_df.sample_set.unique():
            if sample_set not in [self.TEST_SET_NAME, self.OOT_SET_NAME]:
                continue

            logging.log(18, sample_set)
            for y_col in all_decile_df['Y_definition'].unique():
                logging.log(18, y_col)
                cum_bad_rate_dict = {}
                for model_label in all_decile_df['model_label'].unique():
                    the_bool = (all_decile_df.Y_definition==y_col) \
                             & (all_decile_df.model_label==model_label)\
                             & (all_decile_df.sample_set==sample_set)

                    # 在self.generate_decile()的最后，model_label应该已经从存储时用的
                    # 模型名称替换成了用于最终文档展示的全英文更规范的名称，这里double check一下
                    if model_label in self.liftcurve_name_map.keys():
                        model_label = self.liftcurve_name_map[model_label]

                    cum_bad_rate_dict[model_label] = all_decile_df.loc[the_bool, '累积Bad占比'].copy()

                if self.BASE_MODEL_LABEL:
                    pl.plot_lift_curve(cum_bad_rate_dict, self.BASE_MODEL_LABEL,
                                    sample_set, y_col, os.path.join(self.RESULT_PATH, 'figure/lift_curve'))
                else:
                    pl.plot_lift_curve(cum_bad_rate_dict, list(cum_bad_rate_dict.keys())[0],
                                    sample_set, y_col, os.path.join(self.RESULT_PATH, 'figure/lift_curve'))



        # 同一个模型，对比不同set上的lift curve
        for model_label in all_decile_df.model_label.unique():
            if model_label == self.BASE_MODEL_LABEL:
                continue

            logging.log(18, sample_set)
            for y_col in all_decile_df['Y_definition'].unique():
                logging.log(18, y_col)
                cum_bad_rate_dict = {}
                for sample_set in all_decile_df.sample_set.unique():
                    the_bool = (all_decile_df.Y_definition==y_col) \
                             & (all_decile_df.model_label==model_label)\
                             & (all_decile_df.sample_set==sample_set)

                    # 在self.generate_decile()的最后，model_label应该已经从存储时用的
                    # 模型名称替换成了用于最终文档展示的全英文更规范的名称，这里double check一下
                    if model_label in self.liftcurve_name_map.keys():
                        model_label = self.liftcurve_name_map[model_label]

                    cum_bad_rate_dict[sample_set] = all_decile_df.loc[the_bool, '累积Bad占比'].copy()

                pl.plot_lift_curve(cum_bad_rate_dict, self.TRAIN_SET_NAME,
                                model_label, y_col, os.path.join(self.RESULT_PATH, 'figure/lift_curve'))


        logging.info('第五步模型验证：lift curves画图存储于文件夹 %s' % os.path.join(self.RESULT_PATH, 'figure/lift_curve'))



    def swap_analysis(self):
        logging.info('第五步模型验证：swap analysis对比新老模型表现')
        if self.OOT_SET_NAME:
            base_model = self.base_model_score[self.OOT_SET_NAME].to_frame(self.BASE_MODEL_LABEL)

            writer = pd.ExcelWriter(os.path.join(self.RESULT_PATH, '对比模型SWAP_ANALYSIS.xlsx'))

            for model_label in self.model_labels:
                compare_score_dict = load_data_from_pickle(self.DATA_PATH, '模型%s分数.pkl' % model_label)
                compare_label_formal = self.liftcurve_name_map[model_label]
                compare_model = compare_score_dict[self.OOT_SET_NAME]['score'].to_frame(compare_label_formal)

                r_list = []
                for y_col in self.y_cols:
                    data = base_model.merge(compare_model, left_index=True, right_index=True)\
                                     .merge(self.all_Y[[y_col]], left_index=True, right_index=True)
                    data = data.loc[data[y_col].notnull()]
                    result = vp.get_swap_table(data, self.BASE_MODEL_LABEL, compare_label_formal, y_col)
                    r_list.append(result)

                result_for_all_y = pd.concat(r_list)
                result_for_all_y.to_excel(writer, compare_label_formal+'_'+self.OOT_SET_NAME, index=False)

            writer.save()

        logging.info('第五步模型验证：swap analysis结果存储于 %s' % os.path.join(self.RESULT_PATH, '对比模型SWAP_ANALYSIS.xlsx'))



    def score_stability(self):
        logging.info('第五步模型验证：各模型分数稳定性')
        r_list = []
        for model_label in self.model_labels:
            logging.log(18, "Score Stability: model %s" % model_label)
            score_dict = load_data_from_pickle(self.DATA_PATH, '模型%s分数.pkl' % model_label)
            if len(self.backscore_has_perf) > 0:
                score_dict[self.BACKSCORE_SET_NAME]['score'] = score_dict[self.BACKSCORE_SET_NAME]['score'].loc[self.backscore_has_perf].dropna()

            score_psi_result = vp.score_stability(score_dict, self.all_Y,
                                            self.Y_BUILD_NAME, self.TRAIN_SET_NAME,
                                            self.RESULT_PATH, self.liftcurve_name_map[model_label])
            r_list.append(score_psi_result)

        result = pd.concat(r_list)
        result.to_excel(os.path.join(self.RESULT_PATH, 'all_model_score_psi.xlsx'), index=False)
        logging.info('第五步模型验证：各模型分数稳定性PSI结果存储于 %s' % os.path.join(self.RESULT_PATH, 'all_model_score_psi.xlsx'))


    def backscore_approval_rate_analysis(self):
        logging.info('第五步模型验证：通过率和逾期率分析')
        r_list = []
        for model_label in self.model_labels:
            score_dict = load_data_from_pickle(self.DATA_PATH, '模型%s分数.pkl' % model_label)

            model_formal_label = self.liftcurve_name_map[model_label]
            dist_ct = vp.approval_rate_anaysis(score_dict[self.BACKSCORE_SET_NAME]['score'],
                                self.RESULT_PATH, model_formal_label, self.Y_BUILD_NAME)

            pl.plot_backscore_approval_rate(dist_ct, os.path.join(self.RESULT_PATH, 'figure'),
                                        model_formal_label, model_label)
            r_list.append(dist_ct)

        result = pd.concat(r_list)
        result.to_excel(os.path.join(self.RESULT_PATH, 'backscore通过率分析数据.xlsx'), index=False)
        logging.info('第五步模型验证：通过率和逾期率分析结果存储于 %s' % os.path.join(self.RESULT_PATH, 'backscore通过率分析数据.xlsx'))
        logging.info('第五步模型验证：通过率和逾期率分析画图存储于 %s' % os.path.join(self.RESULT_PATH, 'figure'))



class StepSixDeploy(object):
    """
    当模型选择好了之后生成deploy文档和部署文件并上传分数
    """
    def __init__(self, DATA_PATH, RESULT_PATH):
        self.DATA_PATH = DATA_PATH
        self.RESULT_PATH = RESULT_PATH

        if not os.path.exists(os.path.join(self.RESULT_PATH, 'deployment')):
            os.makedirs(os.path.join(self.RESULT_PATH, 'deployment'))

        self.DEPLOY_PATH = os.path.join(self.RESULT_PATH, 'deployment')


    def generate_score(self, model_label, X, new_set_name):
        """
        如果有新样本需要计算分数

        Args:
        model_label (str): 用于存储的区分不同模型的模型名称
        X (pd.DataFrame): 当需要计算分数的模型是逻辑回归模型时，则传入的X为X_cat即分箱好的
            X数据。如果为XGBoost模型时，则需要传入XGBoost预处理好的数据
        new_set_name (str): 需要算分的新的数据集的名称

        Returns:
        新数据集的分数将更新于
        os.path.join(self.DATA_PATH, '模型%s分数.pkl' % model_label)
        """
        score_dict = load_data_from_pickle(self.DATA_PATH, '模型%s分数.pkl' % model_label)

        if 'XGB' not in model_label:
            score_card = pd.read_excel(os.path.join(self.RESULT_PATH, '%s_score_card.xlsx' % model_label))
            varscore, score = mt.Performance().calculate_score_by_scrd(X, score_card)
            score_dict[new_set_name] = {'varscore': varscore, 'score': score}

            save_data_to_pickle(score_dict, self.DATA_PATH, '模型%s分数.pkl' % model_label)
            logging.info("新数据集分数以更新存储于 %s" % os.path.join(self.DATA_PATH, '模型%s分数.pkl' % model_label))

        else:
            model_result = load_data_from_pickle(self.DATA_PATH, '%s模型结果.pkl' % model_label)

            dd_dmatrix = xgb.DMatrix(X)
            p = pd.Series(model_result['model_final'].predict(dd_dmatrix), index=X.index)
            score = p.apply(mt.Performance().p_to_score)
            score_dict[new_set_name] = {'score': score, 'prob': p}

            save_data_to_pickle(score_dict, self.DATA_PATH, '模型%s分数.pkl' % model_label)
            logging.info("新数据集分数以更新存储于 %s" % os.path.join(self.DATA_PATH, '模型%s分数.pkl' % model_label))




    def upload_score_to_database_for_risk(self, model_label, liftcurve_name_map, train_set_name):
        score_dict = load_data_from_pickle(self.DATA_PATH, '模型%s分数.pkl' % model_label)

        all_decile_df = pd.read_excel(os.path.join(self.RESULT_PATH, 'all_decile.xlsx'))
        model_decile = all_decile_df.loc[all_decile_df.model_label==liftcurve_name_map[model_label]].copy()

        # 生成切分和名字替换
        def _process_num(x):
            x = x+1
            if len(str(x)) == 1:
                return '0'+str(x)
            else:
                return str(x)

        def _obtain_bounds(deciles, q, point_bounds_dict, rename_map):
            if q == 10:
                sample_set = train_set_name
            if q == 20:
                sample_set = 'RUNBOOK'
            the_bins = deciles.loc[deciles.sample_set==sample_set, '分箱'].drop_duplicates()
            point_bounds = mt.BinWoe().obtain_boundaries(the_bins)['cut_boundaries']
            dd = the_bins.reset_index().reset_index()
            dd.loc[:, 'new_name'] = dd.apply(lambda row: _process_num(row['level_0']) + str(row['分箱']), 1)
            rename_map['decile%s' % str(q)] = dict(zip(dd['分箱'], dd.new_name))
            point_bounds_dict['decile%s' % str(q)] = point_bounds

        point_bounds_dict = {}
        rename_map = {}
        _obtain_bounds(model_decile, 10, point_bounds_dict, rename_map)
        _obtain_bounds(model_decile, 20, point_bounds_dict, rename_map)

        save_data_to_pickle(point_bounds_dict, self.DATA_PATH, '%s模型分数10等20等分箱边界.pkl' % model_label)
        save_data_to_pickle(rename_map, self.DATA_PATH, '%s模型分箱换名字.pkl' % model_label)

        logging.info("""第六步部署：上传分数至数据库。
        分数切分相关文件存储于：
        1. %s
        2. %s
        """ % (
            os.path.join(self.DATA_PATH, '%s模型分数10等20等分箱边界.pkl' % model_label),
            os.path.join(self.DATA_PATH, '%s模型分箱换名字.pkl' % model_label)
            )
        )

        all_score_df = pd.concat([i['score'] for i in score_dict.values()]).to_frame('score').reset_index()
        all_score_df = all_score_df.drop_duplicates()
        all_score_df.loc[:, 'decile10'] = pd.cut(all_score_df.score,
                                                 point_bounds_dict['decile10'])\
                                            .astype(str)\
                                            .replace(rename_map['decile10'])
        all_score_df.loc[:, 'decile20'] = pd.cut(all_score_df.score,
                                                 point_bounds_dict['decile20'])\
                                            .astype(str)\
                                            .replace(rename_map['decile20'])

        id_name = [i for i in all_score_df.columns if i not in ['score', 'decile10', 'decile20']][0]

        # 建表
        SQL_CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS hive_pro.hdp_data_posp.tmp_%s (
            %s VARCHAR,
            score DOUBLE,
            decile10 VARCHAR,
            decile20 VARCHAR
        )
        """ % (liftcurve_name_map[model_label], id_name)

        presto_upload_data(SQL_CREATE_TABLE)


        SQL_INSERT_TEMP = """
        INSERT INTO hive_pro.hdp_data_posp.tmp_{{ table_name }}
        ({{ id_name }}, score, decile10, decile20)
        VALUES
        {% for i in var_list %}
        {{ i }},
        {% endfor %}
        """

        all_score_df[id_name] = all_score_df[id_name].astype(str)
        all_score_df['score'] = all_score_df['score'].astype(float)

        num_lines = 5000
        upload_status = {}
        for line in range(0, len(all_score_df), num_lines):
            logging.log(18, "分数上传数据库：行数%s:%s" % (line, (line+num_lines)))
            var_list =[]
            for index, row in all_score_df.iloc[line:line+num_lines].iterrows():
                var_list.append(tuple(row))

            SQL_UPLOAD = Template(SQL_INSERT_TEMP).render(var_list=var_list,
                                                          table_name=liftcurve_name_map[model_label],
                                                          id_name=id_name).replace('\n', '')
            last_comma_position = [pos for pos, char in enumerate(SQL_UPLOAD) if char == ','][-1]

            status = presto_upload_data(SQL_UPLOAD[:last_comma_position])

            if status == 'failure':
                logging.error("分数上传数据库：行数%s:%s 失败" % (line, (line+num_lines)))
                upload_status[line] = {'SQL': SQL_UPLOAD, 'status': status}
            else:
                logging.log(18, "分数上传数据库：行数%s:%s 成功" % (line, (line+num_lines)))

        if len(upload_status) > 0:
            save_data_to_pickle(upload_status, self.DATA_PATH, '上传数据库失败%s.pkl' % model_label)
            logging.log(18, '上传失败SQL存储于%s' % os.path.join(self.DATA_PATH, '上传数据库失败%s.pkl' % model_label))

        logging.info("第六步部署：上传分数至数据库。数据上传至表格：hive_pro.hdp_data_posp.tmp_%s" % liftcurve_name_map[model_label])


    def generate_scorecard_deployment_documents(self, model_label, live_abbr,
                                                y_col, train_set_name,
                                                eda_file_name,
                                                coarse_classing_rebin_spec,
                                                production_modelName, product_name,
                                                liftcurve_name_map,
                                                production_name_map={}):

        """
        生成逻辑回归评分卡部署Excel文件，线上部署json文件，线上部署测试用例

        Args
        model_label (str): 模型名称
        live_abbr (str): 上线后模型输出打分字段尾缀名称
        y_col (str): 建模Y定义名称
        train_set_name (str): 建模样本名称
        eda_file_name (str): EDA表格存储文件名称
        coarse_classing_rebin_spec (dict): 建模分箱边界文件
        production_modelName (str): 需要部署的模型在线上部署时所在的条用modelName，取值问部署人员
        product_name (str): 中间层表格中的product_name取值。传入这个模型所部署的产品名
        liftcurve_name_map (dict): 画lift curve的时候原始model_label转换成图中的
            显示名称，因为画图会将中文显示成方格，所以转换的值需全为英文。取名时请注意规范一致
            性和可读性和理解性，因为这个是会放到最终报告中的。如果有用于对比表现的老模型，也
            需要包含。key值为建模时各模型对应的model_label, value值为规范刻度和解释性较好
            的全英文正式模型名称
        production_name_map (dict): 当有指标英文线上的名字和线下建模时不一样时传入。key=建模时英文名
            value=线上英文名。 default={} 默认建模时和线上没有命名不一致的情况
        """
        eda_table = pd.read_excel(os.path.join(self.RESULT_PATH, eda_file_name))

        score_card = pd.read_excel(os.path.join(self.RESULT_PATH, '%s_score_card.xlsx' % model_label))
        score_card_production = score_card.rename(columns={'指标英文': '中间层指标名称', '变量打分': '打分'})
        score_card_production['中间层指标名称'] = score_card_production['中间层指标名称'].replace(production_name_map)
        score_card_production.insert(2, '输出打分指标名称',
                    score_card_production['中间层指标名称'].apply(lambda x: 'mlr_' + x + '_scrd_' + live_abbr))

        score_card_production.loc[:, '输出打分指标名称'] = score_card_production.loc[:, '输出打分指标名称']\
                                                            .replace('mlr_intercept_scrd_'+live_abbr, 'mlr_const_scrd_'+live_abbr)

        score_card_production.loc[score_card_production['中间层指标名称']=='intercept', '指标中文'] = '截距分'
        score_card_production.loc[score_card_production['中间层指标名称']=='intercept', '中间层指标名称'] = None

        score_card_production = score_card_production.append({'输出打分指标名称': 'mlr_creditscore_scrd_'+live_abbr}, ignore_index=True)
        score_card_production = score_card_production.append({'输出打分指标名称': 'mlr_prob_scrd_'+live_abbr}, ignore_index=True)
        score_card_production.insert(score_card_production.shape[1], '是否手动调整', None)
        score_card_production.insert(score_card_production.shape[1], 'backscore分布占比', None)
        score_card_production.insert(score_card_production.shape[1], '基准psi', None)
        score_card_production.insert(score_card_production.shape[1], 'psi预警delta', None)


        all_decile_df = pd.read_excel(os.path.join(self.RESULT_PATH, 'all_decile.xlsx'))
        the_bool = ((all_decile_df.model_label==liftcurve_name_map[model_label]) &
                    (all_decile_df.Y_definition==y_col) &
                    (all_decile_df.sample_set==train_set_name))
        model_decile = all_decile_df.loc[the_bool].copy()

        selected_variables = [i for i in score_card['指标英文'].unique() if i != 'intercept']
        model_eda = eda_table.loc[eda_table['指标英文'].isin(selected_variables)]

        writer = pd.ExcelWriter(os.path.join(self.DEPLOY_PATH, '%s部署评分卡.xlsx' % model_label))
        score_card_production.to_excel(writer, '2_模型评分卡', index=False)
        model_decile.to_excel(writer, '3_模型decile', index=False)
        model_eda.to_excel(writer, '4_模型EDA', index=False)
        writer.save()
        logging.info("""第六步部署：生成逻辑回归评分卡部署文档。
        1. 模型部署Excel文档存储于：%s
        2. 需添加『0_文档修订记录』、『1_信息总览』页面。详见其他正式部署文档文件。并存储于『/Seafile/模型共享/模型部署文档/』相应文件夹中
        """ % os.path.join(self.DEPLOY_PATH, '%s部署评分卡.xlsx' % model_label))

        bin_to_score_json = mu.process_bin_to_score(score_card)
        for original, new_name in production_name_map.items():
            coarse_classing_rebin_spec[new_name] = coarse_classing_rebin_spec.pop(original)
            bin_to_score_json[new_name] = bin_to_score_json.pop(original)

        rebin_spec_json = mu.process_rebin_spec(coarse_classing_rebin_spec, score_card, selected_variables)

        save_data_to_json(rebin_spec_json, self.DEPLOY_PATH, '%s_selected_rebin_spec.json' % live_abbr)
        save_data_to_json(bin_to_score_json, self.DEPLOY_PATH, '%s_bin_to_score.json' % live_abbr)
        logging.info("""第六步部署：生成逻辑回归评分卡部署文档。
        线上部署配置文件存储于%s路径下
        1. %s
        2. %s
        """ % (self.DEPLOY_PATH,
               '%s_selected_rebin_spec.json' % live_abbr,
               '%s_bin_to_score.json' % live_abbr
              ))


        writer = pd.ExcelWriter(os.path.join(self.DEPLOY_PATH, '%stest_cases.xlsx' % model_label))
        test_case = mu.produce_http_test_case(score_card_production, production_modelName, product_name)
        test_case.to_excel(writer, model_label, index=False)
        writer.save()
        logging.info("""第六步部署：生成逻辑回归评分卡部署文档。
        线上部署测试用例存储于 %s
        """ % os.path.join(self.DEPLOY_PATH, '%stest_cases.xlsx' % model_label))


    def generate_xgb_deployment_documents(self, model_label, live_abbr, y_col,
                                      train_set_name, eda_file_name, var_dict,
                                      model_label_version,
                                      liftcurve_name_map, production_name_map={}):
        """
        生成XGBoost部署文档
        """
        eda_table = pd.read_excel(os.path.join(self.RESULT_PATH, eda_file_name))
        model_spec = load_data_from_pickle(self.DATA_PATH, '%s_XGBoost输出数据和分箱明细.pkl' % model_label_version)
        rebin_spec = model_spec['auto_rebin_spec']
        bin_to_label = model_spec['bin_to_label']
        dummy_var_name_map = model_spec['dummy_var_name_map']

        xgb_importance_score = pd.read_excel(os.path.join(self.RESULT_PATH, '%s模型变量重要性排序.xlsx' % model_label))
        xgb_importance_score = xgb_importance_score.rename(columns={'feature': 'XGB衍生入模名称', 'fscore': '指标用于Split数据的数量', 'imp_pct': '指标用于Split数据的数量占比'})

        xgb_importance_score[['XGB变量转换类型', '中间层指标名称']] = xgb_importance_score['XGB衍生入模名称']\
                            .apply(lambda x: pd.Series(mt.BinWoe().xgboost_obtain_raw_variable(x, var_dict)))

        xgb_importance_score['建模时指标名称'] = xgb_importance_score['中间层指标名称'].copy()
        xgb_importance_score['建模时XGB衍生入模名称'] = xgb_importance_score['XGB衍生入模名称'].copy()
        for original, new_name in production_name_map.items():
            a = xgb_importance_score.loc[xgb_importance_score['建模时指标名称']==original, '建模时XGB衍生入模名称']\
                                    .apply(lambda x: x.replace(original, new_name))
            xgb_importance_score.loc[xgb_importance_score['建模时指标名称']==original, 'XGB衍生入模名称'] = a
            if original in rebin_spec:
                rebin_spec[new_name] = rebin_spec.pop(original)
            if original in bin_to_label:
                bin_to_label[new_name] = bin_to_label.pop(original)

        xgb_importance_score['中间层指标名称'] = xgb_importance_score['中间层指标名称'].replace(production_name_map)

        xgb_importance_score = var_dict[['数据源', '指标英文', '指标中文', '数据类型']]\
                                    .rename(columns={'指标英文':'中间层指标名称'})\
                                    .merge(xgb_importance_score, on='中间层指标名称', how='right')

        xgb_importance_score.insert(5, '输出打分指标名称',
                    xgb_importance_score['XGB衍生入模名称'].apply(lambda x: 'mlr_' + str(x) + '_xgb_' + live_abbr))


        xgb_importance_score = xgb_importance_score.append({'输出打分指标名称': 'mlr_creditscore_xgb_'+live_abbr}, ignore_index=True)
        xgb_importance_score = xgb_importance_score.append({'输出打分指标名称': 'mlr_prob_xgb_'+live_abbr}, ignore_index=True)

        all_decile_df = pd.read_excel(os.path.join(self.RESULT_PATH, 'all_decile.xlsx'))
        the_bool = ((all_decile_df.model_label==liftcurve_name_map[model_label]) &
                    (all_decile_df.Y_definition==y_col) &
                    (all_decile_df.sample_set==train_set_name))
        model_decile = all_decile_df.loc[the_bool].copy()

        selected_variables = xgb_importance_score['建模时指标名称'].unique()
        model_eda = eda_table.loc[eda_table['指标英文'].isin(selected_variables)].copy()
        model_eda['指标英文'] = model_eda['指标英文'].replace(production_name_map)

        writer = pd.ExcelWriter(os.path.join(self.DEPLOY_PATH, '%s部署文档.xlsx' % model_label))
        xgb_importance_score.to_excel(writer, '2_模型变量重要性排序', index=False)
        model_decile.to_excel(writer, '3_模型decile', index=False)
        model_eda.to_excel(writer, '4_模型EDA', index=False)
        writer.save()
        logging.info("""第六步部署：生成XGBoost部署文档。
        1. 模型部署Excel文档存储于：%s
        2. 需添加『0_文档修订记录』、『1_信息总览』页面。详见其他正式部署文档文件。并存储于『/Seafile/模型共享/模型部署文档/』相应文件夹中
        """ % os.path.join(self.DEPLOY_PATH, '%s部署文档.xlsx' % model_label))


        model_result = load_data_from_pickle(self.DATA_PATH, '%s模型结果.pkl' % model_label)
        derive_name_map = dict(zip(xgb_importance_score['建模时XGB衍生入模名称'], xgb_importance_score['XGB衍生入模名称']))
        xgbmodel = model_result['model_final']
        xgbmodel.__dict__['feature_names'] = [derive_name_map[i] for i in xgbmodel.__dict__['feature_names']]

        num_variables = list(xgb_importance_score.loc[xgb_importance_score['XGB变量转换类型']=='num_vars_origin', '中间层指标名称'].unique())
        bin_variables = list(xgb_importance_score.loc[xgb_importance_score['XGB变量转换类型']=='bin_vars', '中间层指标名称'].unique())

        rebin_spec_json = mu.process_rebin_spec(rebin_spec, var_dict, num_variables+bin_variables)
        bin_to_label = {k:v for k,v in bin_to_label.items() if k in bin_variables}

        var_transform_method = {}
        var_transform_method['num_vars_origin'] = num_variables
        var_transform_method['bin_vars'] = bin_variables
        var_transform_method['dummy_vars'] = {}
        dummy_vars_df = xgb_importance_score.loc[xgb_importance_score['XGB变量转换类型']=='dummy_vars'].copy()
        dummy_vars_df.loc[:, "dummy原始名称"] = dummy_vars_df['建模时XGB衍生入模名称'].apply(lambda x: dummy_var_name_map[x])
        dummy_vars_df.loc[:, 'dummy原始对应分类'] = dummy_vars_df.loc[:, "dummy原始名称"].apply(lambda x: x.split('DUMMY')[-1])
        for original_variable in dummy_vars_df['中间层指标名称'].unique():
            cat_list = list(dummy_vars_df.loc[dummy_vars_df['中间层指标名称']==original_variable, 'dummy原始对应分类'].unique())
            var_transform_method['dummy_vars'][original_variable] = cat_list



        save_data_to_json(rebin_spec_json, self.DEPLOY_PATH, '%s_selected_rebin_spec.json' % live_abbr)
        save_data_to_json(bin_to_label, self.DEPLOY_PATH, '%s_bin_to_label.json' % live_abbr)
        save_data_to_python2_pickle(xgbmodel, self.DEPLOY_PATH, '%s_xgbmodel.pkl' % live_abbr)
        save_data_to_json(var_transform_method, self.DEPLOY_PATH, '%s_var_transform_method.json' % live_abbr)

        logging.info("""第六步部署：生成XGBoost部署文档。
        线上部署配置文件存储于%s路径下
        1. %s
        2. %s
        3. %s
        4. %s
        """ % (self.DEPLOY_PATH,
               '%s_selected_rebin_spec.json' % live_abbr,
               '%sbin_to_label.json' % live_abbr,
               '%s_var_transform_method.json' % live_abbr,
               '%s_xgbmodel.pkl' % live_abbr,
              ))
