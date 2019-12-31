#!/usr/bin/env python

################################################################
#S1: 导包, 设置文件访问路径
################################################################
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
#python3种reload函数不能直接使用
from imp import reload
from jinja2 import Template
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt

# 更改为自己的路径,或者直接在环境变量中添加
sys.path.append('/Users/ying.xie/Documents/repositories/sublime_projects/ge/genie_xy')
from utils3.data_io_utils import *
import utils3.misc_utils as mu
import utils3.summary_statistics as ss
import utils3.metrics as mt
import utils3.feature_selection as fs
import utils3.plotting as pl
import utils3.modeling as ml

# 设置绘图风格, 不喜欢seaborn风格可将其注释
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn') # pretty matplotlib plots
plt.rcParams['figure.figsize'] = (10, 8)

Root_path = '/Users/ying.xie/Documents/repositories/sublime_projects/ge'
Data_path = '/Users/ying.xie/Documents/repositories/sublime_projects/ge/01_data'
Result_path = '/Users/ying.xie/Documents/repositories/sublime_projects/ge/02_result'


################################################################
#S2: 读取数据字典, 必要列['指标英文', '数据源', '指标中文', '指标类型']
################################################################
var_dict = pd.read_excel(os.path.join(Root_path, 'var_dict.xlsx'))


################################################################
#S3: 读取数据集, 定义全局变量名
################################################################
model_data_origin = load_data_from_pickle(Data_path,'all_x_y_1227_v6_0425.pkl')
model_data_origin.shape

"""
3.1 将代表 y的变量名单与代表客户标识的主键id 单独定义处理, 方便后续一致性
"""
y_col_name = 'Y'
x_index_name = 'applyid'
save_label = 'all'
app_month = 'app_month'
model_label = 'demo_xgb_v2'

"""
3.2 画好坏分布图, 第一次可将save设置为True, 此后不需要再保存文件, 可移除该参数
"""
plot_total_distribution(model_data_origin[y_col_name], os.path.join(Result_path, 'figure'), save=True)


"""
3.3 整理X,y数据. 此时选出的X数据中就只有字典里边的字段了, 且X, y均为dataframe
"""
import copy
model_data_final = copy.deepcopy(model_data_origin)
# 如果index不是主键id, 这里设置, 是则跳过这一步
model_data_final.set_index(x_index_name, inplace=True)
model_data_final.index

X = model_data_final[var_dict['指标英文'].unique()]
y = model_data_final[[y_col_name]]
y.shape, X.shape

X.isnull().sum().sum()



################################################################
#S4: 填充缺失值, 优化数据集大小, 并生成新的变量字典(新增一列 数据类型)
################################################################
X = mu.process_missing(X, var_dict, known_missing={-9999999: -8887})
X.replace(np.inf,9999,inplace=True)
X.isnull().sum().sum()

# 优化数据集, 生成新的变量字典
X, var_dict = mu.optimal_mem(X, var_dict, verbose=True)
# 确认数据类型已经正确设置
len(var_dict[var_dict['数据类型'].isnull()]['指标英文'])
# 保存新的变量字典
var_dict.to_excel(os.path.join(Root_path, 'var_dict_v2.xlsx'))



################################################################
#S5: EDA 通过这一步可以得到数据的一些描述性统计数据
################################################################
ss.eda(X, var_dict, useless_vars=[], exempt_vars=[], data_path=Result_path, save_label=save_label)

# # 筛掉Summary excluded变量进行综合排序
summary = pd.read_excel(os.path.join(Result_path, '{}_variables_summary.xlsx'.format(save_label)), encoding='utf-8')
kept = summary.loc[pd.isnull(summary.exclusion_reason), '指标英文'].tolist()
# 填充缺失值时如果有-8888，X的index会被打乱
X = pd.merge(X,y,left_index=True,right_index=True)[kept]
y = pd.merge(X,y,left_index=True,right_index=True)['Y']



################################################################
#S6: 将字符串变量进行labelencode, 保存转换映射字典
################################################################
X, all_encoding_map = mu.encode_categorical_by_all(X, var_dict)
save_data_to_pickle(all_encoding_map, Result_path, 'all_encoding_map.pkl')




################################################################
#S7: 划分训练集和测试集
################################################################
from sklearn.model_selection import train_test_split
train_apply, test_apply = train_test_split(model_data_final.index, test_size=0.30, random_state=43)

model_data_final.loc[:, 'sample_set'] = np.where(model_data_final.index.isin(train_apply), 'train', 'test')

X_train = X.loc[model_data_final.index.isin(list(model_data_final.loc[model_data_final.sample_set=='train'].index))]
X_test = X.loc[model_data_final.index.isin(list(model_data_final.loc[model_data_final.sample_set=='test'].index))]
y_train = y.loc[model_data_final.index.isin(list(model_data_final.loc[model_data_final.sample_set=='train'].index))]
y_test = y.loc[model_data_final.index.isin(list(model_data_final.loc[model_data_final.sample_set=='test'].index))]

# ## 仔细查看train，test数据确保正确
X_train.shape,y_train.shape,X_test.shape,y_test.shape
sum(X_train.index != y_train.index),sum(X_test.index != y_test.index)




################################################################
# S8: 各种算法的overall_ranking
################################################################
args_dict = {
    'random_forest': {
        'grid_search': False,#选择了True则会进行网格筛选速度会比较慢
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
    'xgboost'
]
fs_obj = fs.FeatureSelection()
# 使用日志在result_path中可以通过日志观察overall_ranking进行到哪一步了
logging.basicConfig(filename=os.path.join(Result_path, 'test_log.log'), level=logging.INFO, filemode='w')
LOG = logging.getLogger(__name__)

"""
8.1 woe转换与分箱
# X_cat_train:变量分箱以后的对应的箱
# X_transformed:变量分箱以后的对应的箱的woe值
# woe_iv_df:变量细分箱结果
# rebin_spec:变量对应的分箱节点
# ranking_result:每个变量通过算法得到的所在排位
# 这一步时间比较长请大家耐心
# 在XGboost中不建议分箱太细，这样极其容易过拟合，num_max_bins建议用5分箱
"""

X_cat_train, X_transformed, woe_iv_df, rebin_spec, ranking_result = fs_obj.overall_ranking(X_train, y_train, var_dict, args_dict, methods, num_max_bins=5)
# 将编码前的原始变量添加到woe_iv_df中
mt.add_origin_categorical_val(woe_iv_df, all_encoding_map)
# 保存
save_data_to_pickle(X_cat_train,Result_path,'X_cat_train.pkl')
save_data_to_pickle(woe_iv_df,Result_path,'woe_iv_df.pkl')
save_data_to_pickle(rebin_spec,Result_path,'rebin_spec.pkl')
save_data_to_pickle(ranking_result,Result_path,'ranking_result.pkl')
save_data_to_pickle(X_transformed,Result_path,'ranking_result.pkl')



################################################################
#S8: 相关性筛变量, 留下整体排名靠前的变量
################################################################
n = 100
top_n = ranking_result.sort_values('overall_rank')[u'指标英文'].iloc[:n].tolist()
# 决定从top_n中剔除的字段名称; 决定模型一定要有的字段，即使可能不显著
exclusion, start_from = [], []
# 通过综合排序选中的
selected = list(set(top_n) - set(exclusion))
# 将数据分箱后并转换woe值
X_transformed_train = X_transformed[selected]

corr_threshold = 0.7 # 可调
vif_threshold = 10 # 可调
#相关性筛选
vif_result = fs.Colinearity(X_transformed_train, Result_path)
# 这一步时间比较长请大家耐心
vif_result.run(ranking_result)

# 相关性分析删除的变量
exclusion_cols = vif_result.corr_exclude_vars + vif_result.vif_dropvars
exclusion_cols

# # 调整粗分箱
selected = list(set(selected) - set(exclusion_cols))
selected





################################################################
#S9: 适用于逻辑回归的调分箱, 衍生bin_num, dummy变量的处理方式
################################################################

# ## 训练集按照新的rebin_spec进行分箱
#apply new bin cutting to the data
bin_obj = mt.BinWoe()
new_X = X_train[selected]
X_cat_train = bin_obj.convert_to_category(new_X, var_dict, rebin_spec)
woe_iv_df_coarse = bin_obj.calculate_woe_all(X_cat_train, y_train, var_dict, rebin_spec)
# 字符型变量编码前的原始值写入粗分箱文件中
mt.add_origin_categorical_val(woe_iv_df_coarse, all_encoding_map)
# ##将woe_iv_df_coarse存储出去，为了部署监控使用
save_data_to_pickle(woe_iv_df_coarse,Result_path,'woe_iv_df_coarse.pkl')

# ## 测试集按照新的rebin_spec进行分箱
new_X = X_test[selected]
X_cat_test = bin_obj.convert_to_category(new_X, var_dict, rebin_spec)

# # 按月查看训练集分箱以后的分布以及逾期率
# ## 将分好箱的数据与原数据中的y以及时间标识合并

X_cat_train_with_y_appmon = pd.merge(X_cat_train, model_data_final[[y_col_name,app_month]]                                     ,left_index=True,right_index=True)
# 这一步按月统计了变量的分布以及缺失率，这一步可以将字段先存储出去，观察不合格的变量，在下一步可以提前删除
var_dist_badRate_by_time = ss.get_badRate_and_dist_by_time(X_cat_train_with_y_appmon, woe_iv_df_coarse, selected, 'app_month', 'Y')
# 建议到这一步的时候讲数据存储出去，观察一下
var_dist_badRate_by_time.to_excel(os.path.join(Result_path,'var_dist_badRate_by_time.xlsx'))
# 通过观察var_dist_badRate_by_time.xlsx文件决定将按时间分布不稳定的变量删除
exclusion_vars_by_time_dist_and_badrate = ['apollo_hisMaxOverdueDays','fid2AppDays']

# # 将变量转化为XGBoost需要的形式
# ## 保存最终留下的字段
selected_final = list(set(selected) - set(exclusion_vars_by_time_dist_and_badrate))
X_cat_train_final = X_cat_train[selected_final]
X_train_final = X_train[selected_final]

bin_obj = mt.BinWoe()
# X_train_xgboost:xgboost最终衍生完成的所有变量的集合
# rebin_spec_bin_adjusted_final:最终的分箱文件
# bin_to_label:将分箱转化为标签的dict文件
# dummy_var_name_map:转化为dummy的dict文件
X_train_xgboost, rebin_spec_bin_adjusted_final, bin_to_label, dummy_var_name_map = bin_obj.xgboost_data_derive(X_train_final, y_train, var_dict,rebin_spec_adjusted=rebin_spec, verbose=True)

# ## 将测试集进行XGboost转转换
X_test_final = X_test[selected_final]
X_cat_test_final = X_cat_test[selected_final]

X_test_xgboost = mt.BinWoe().apply_xgboost_data_derive(X_test_final,
                                var_dict, rebin_spec_bin_adjusted_final, bin_to_label,\
                                                       dummy_var_name_map=dummy_var_name_map)


# ## 将训练集的调整后的分箱数据进行dummy并与之前衍生的数据进行合并（这一步可尝试做，现在看来增益不太大）
X_train_xgboost_with_bin_dummy = bin_obj.convert_xgboost_rebins_to_dummy(X_train_xgboost,X_cat_train_final,dummy_var_name_map)
X_train_xgboost_with_bin_dummy.shape

# is_apply = True,将原先train上边的映射逻辑应用
X_test_xgboost_with_bin_dummy = bin_obj.convert_xgboost_rebins_to_dummy(X_test_xgboost,X_cat_test_final,dummy_var_name_map,is_apply=True)
X_test_xgboost_with_bin_dummy.shape

# ## 在一些字符型变量中极其容易出现训练集和测试集中从未出现的取值，这时一定要小心，必要时刻可以将特殊取值进行取舍
set(X_train_xgboost_with_bin_dummy.columns) - set(X_test_xgboost_with_bin_dummy.columns)
set(X_test_xgboost_with_bin_dummy.columns) - set(X_train_xgboost_with_bin_dummy.columns)

#X_train_xgboost_with_bin_dummy.drop(['fstDrawdown2NowDays_bin_dummy4',
# 'lstDrawdown2AppDays_bin_dummy4',
# 'lstDrawdown2NowDays_bin_dummy4',
# 'maxCuroverdueday_bin_dummy2',
# 'meanDrawdownAmtHist_bin_dummy4',
# 'meanDrawdownHourHist_bin_dummy4',
# 'meanTenorHist_bin_dummy4',
# 'ratio_ob_bin_dummy2',
# 'recent_180D_tongdunBlackDecisionDUMMY2',
# 'recent_180D_tongdunBlackDecision_bin_dummy2',
# 'riskGradeDUMMY4',
# 'riskGradeDUMMY5',
# 'riskGradeDUMMY6'],axis=1,inplace=True)
#X_test_xgboost_with_bin_dummy.drop(['riskGradeDUMMY2', 'riskGrade_-9999'],axis=1,inplace=True)
X_train_xgboost_with_bin_dummy.shape,X_test_xgboost_with_bin_dummy.shape

# ## 经过理论以及实践验证已经发现bin_num以及bin_dummy变量在模型中几乎没有增益，因此建议删除这些变量，省的浪费训练时间
X_train_xgboost_with_bin_dummy.columns
# bin_dummy相关的字段，这一步如果想留下可以保留，因为有可能会有效果，在不同的模型中起到的效果不一，总体提升有限
# 如果想要衍生相关的变量也可以向中间层提出需求
bin_dummy_list = [i for i in X_train_xgboost_with_bin_dummy.columns if 'bin_dummy' in i ]
# binnum直接删除，完全违背xgboost理论
bin_num_list = [i for i in X_train_xgboost_with_bin_dummy.columns if 'binnum' in i ]
X_train_xgboost_with_bin_dummy.drop(bin_num_list + bin_dummy_list,axis=1,inplace=True)
X_test_xgboost_with_bin_dummy.drop(bin_num_list + bin_dummy_list,axis=1,inplace=True)
X_train_xgboost_with_bin_dummy.shape,X_test_xgboost_with_bin_dummy.shape
# 其中demo_xgb这个label很重要之后还会用到
save_data_to_pickle({
            'X_train_xgboost': X_train_xgboost_with_bin_dummy,
            'X_test_xgboost': X_test_xgboost_with_bin_dummy,
            'rebin_spec': rebin_spec_bin_adjusted_final,
            'bin_to_label': bin_to_label,
            'dummy_var_name_map': dummy_var_name_map}, Data_path,\
        '%s_XGBoost输出数据和分箱明细.pkl' % 'demo_xgb')

################################################################
#从S9至此, 可以忽略不执行
################################################################


################################################################
#S10: 模型评估
################################################################

"""
10.1 XGboost参数设置
"""
sample_split_summary = ss.sample_split_summary(y_train,y_test)
sample_split_summary

# 调参方法，可自行选择
xgbparams_selection_method = ['XGBExp', 'XGBRandom','XGBGrid','XGBHyperopt']
# 超参调节
import utils3.modeling as ml
import utils3.feature_selection as fs
reload(ml)
reload(fs)
from hyperopt import hp
from hyperopt.pyll.stochastic import sample as hp_sample

space = {'max_depth': hp.randint('max_depth', 5), # 树的深度
         'n_estimators': hp.randint('n_estimators', 40), # CART树的数量
         'scale_pos_weight': hp.randint('scale_pos_weight', 6), # bad/good, 不给出时, 默认为1, 即坏:好=1:1
         'subsample':  hp.randint('subsample', 5),
         'min_child_weight': hp.randint('min_child_weight', 6), # 所有树的叶子最小权重和(损失函数的二阶导数和), 大于阈值时继续分割
         'learning_rate': hp.uniform('learning_rate', 1e-3, 5e-1), # 学习率, 越小训练越慢
         'gamma': hp.uniform('gamma', 1e-3, 5e-1), # 树生长临界值, 小于该值, 树停止生长
         'reg_alpha': hp.uniform('reg_alpha', 1e-3, 5e-1), # L1正则项
         'colsample_bytree':  hp.uniform('colsample_bytree', 0.2, 1), #每棵树训练时使用的特征数量
        }

# 经验参数
xgb_param_experience = {
            'max_depth': 2,
            'min_samples_leaf ': 200,
             'eta': 0.1,
             'objective': 'binary:logistic',
             'subsample': 0.8,
             'colsample_bytree': 0.8,
             'gamma': 0,
             'silent': 0,
             'eval_metric':'auc'
        }
# 随机调参时刻的相关参数
xgb_params_range = {
            'learning_rate': [0.03, 0.05, 0.1, 0.2],
            'max_depth': [2, 3, 4, 5],
            'gamma': [0, 0.05, 0.1],
            'min_child_weight':[1, 3],
            'subsample': np.linspace(0.4, 0.7, 3),
            'colsample_bytree': np.linspace(0.4, 0.7, 3)
        }

"""
10.2 拟合模型
"""
xgbmodel_obj = ml.XGBModel(var_dict, Data_path, Result_path, model_label, y_train, y_test=y_test,
                         params_selection_method=xgbparams_selection_method,
                         x_train=X_train[selected], x_test=X_test[selected],
                         use_origin=True)
xgbmodel_obj.run(xgb_params_range, xgb_param_experience, space=space, xgb_select=False)
xgbparams_selection_method

# ### 训练集和测试集拟合结果 
plot_obj = ml.LogisticModel(var_dict=var_dict, y_train=y_train,y_test=y_test)

"""
10.3 查看XGBHyperopt（超参）模型结果
"""
xgbparams_selection_method
model_result_hyperopt = load_data_from_pickle(Data_path,'%s模型结果.pkl' % (model_label+'XGBHyperopt'))
plot_obj.plot_for_the_model(Result_path, model_result_hyperopt, with_test=True)

# #### 查看XGBHyperopt（超参）对应的每一颗决策树的决策点
model_result_hyperopt['model_final'].dump_model(Result_path+'/model_result_hyperopt_trees.txt')

# #### 查看XGBHyperopt（超参）每一颗决策树对应的information和population
train_model_result_hyperopt_leafs = ss.get_xgboost_tree_leaf_dist_and_badRate(model_result_hyperopt,X_train[selected],y_train,y_col_name)
train_model_result_hyperopt_leafs.head(10)

"""
10.4 XGBHyperopt模型decile
"""
train_XGBHyperopt_score = model_result_hyperopt['p_train'].apply(mt.Performance().p_to_score)
test_XGBHyperopt_score = model_result_hyperopt['p_test'].apply(mt.Performance().p_to_score)

# 训练集
ks_decile_XGBHyperopt_train = mt.Performance().calculate_ks_by_decile(train_XGBHyperopt_score, np.array(y_train), 'decile', q=10)
ks_decile_XGBHyperopt_train

# 测试集
point_bounds_XGBHyperopt = mt.BinWoe().obtain_boundaries(ks_decile_XGBHyperopt_train[u'分箱'])['cut_boundaries']
ks_decile_XGBHyperopt_test = mt.Performance().calculate_ks_by_decile(test_XGBHyperopt_score, np.array(y_test), 'decile', manual_cut_bounds=point_bounds_XGBHyperopt)
ks_decile_XGBHyperopt_test

# 生成XGBHyperopt模型runbook
run_XGBHyperopt_book = mt.Performance().calculate_ks_by_decile(pd.concat([train_XGBHyperopt_score,test_XGBHyperopt_score],join='outer'), np.array(pd.concat([y_train,y_test,],join='outer')), 'decile', q=20)
run_XGBHyperopt_book


"""
10.5 查看XGBExp（经验参数）模型结果
"""
model_result_exp = load_data_from_pickle(Data_path, '%s模型结果.pkl' % (model_label+'XGBExp'))
plot_obj.plot_for_the_model(Result_path, model_result_exp, with_test=True)


# #### 查看XGBExp（经验参数）对应的每一颗决策树的决策点
model_result_exp['model_final'].dump_model(Result_path+'/model_result_exp_trees.txt')
# #### 查看XGBExp（经验参数）每一颗决策树对应的information和population
train_model_result_exp_leafs = ss.get_xgboost_tree_leaf_dist_and_badRate(model_result_exp,X_train[selected],y_train,y_col_name)
train_model_result_exp_leafs.head()


"""
10.6 查看XGBExp模型decile
"""
train_XGBExp_score = model_result_exp['p_train'].apply(mt.Performance().p_to_score)
test_XGBExp_score = model_result_exp['p_test'].apply(mt.Performance().p_to_score)

# 训练集
ks_decile_XGBExp_train = mt.Performance().calculate_ks_by_decile(train_XGBExp_score, np.array(y_train), 'decile', q=10)
ks_decile_XGBExp_train

# 测试集
point_bounds_XGBExp = mt.BinWoe().obtain_boundaries(ks_decile_XGBExp_train[u'分箱'])['cut_boundaries']
ks_decile_XGBExp_test = mt.Performance().calculate_ks_by_decile(test_XGBExp_score, np.array(y_test), 'decile',manual_cut_bounds=point_bounds_XGBExp)
ks_decile_XGBExp_test

# 生成XGBExp模型runbook
run_XGBExp_book = mt.Performance().calculate_ks_by_decile(pd.concat([train_XGBExp_score,test_XGBExp_score],join='outer'), np.array(pd.concat([y_train,y_test,],join='outer')), 'decile',q=20)
run_XGBExp_book

"""
10.7 查看XGBRandom（随机参数）模型结果
"""
model_result_random = load_data_from_pickle(Data_path, '%s模型结果.pkl' % (model_label+'XGBRandom'))
plot_obj.plot_for_the_model(Result_path, model_result_random, with_test=True)

# 查看XGBRandom（随机参数）对应的每一颗决策树的决策点
model_result_random['model_final'].dump_model(Result_path+'/model_result_random_trees.txt')
# 查看XGBRandom（随机参数）每一颗决策树对应的information和population, 第二个参数可以是X_train_xgboost_with_bin_dummy/X_train
train_model_result_random_leafs = ss.get_xgboost_tree_leaf_dist_and_badRate(model_result_random,X_train[selected],y_train,y_col_name)
train_model_result_random_leafs.head(10)

"""
10.8 XGBRandom模型decile
"""
train_XGBRandom_score = model_result_random['p_train'].apply(mt.Performance().p_to_score)
test_XGBRandom_score = model_result_random['p_test'].apply(mt.Performance().p_to_score)

# 训练集
ks_decile_XGBRandom_train = mt.Performance().calculate_ks_by_decile(train_XGBRandom_score, np.array(y_train), 'decile', q=10)
ks_decile_XGBRandom_train

# 测试集
point_bounds_XGBRandom = mt.BinWoe().obtain_boundaries(ks_decile_XGBRandom_train[u'分箱'])['cut_boundaries']
ks_decile_XGBRandom_test = mt.Performance().calculate_ks_by_decile(test_XGBRandom_score, np.array(y_test), 'decile', manual_cut_bounds=point_bounds_XGBRandom)
ks_decile_XGBRandom_test

# 生成XGBRandom模型runbook
run_XGBRandom_book = mt.Performance().calculate_ks_by_decile(pd.concat([train_XGBRandom_score,test_XGBRandom_score],join='outer'), np.array(pd.concat([y_train,y_test,],join='outer')), 'decile', q=20)
run_XGBRandom_book

"""
10.9 查看XGBGrid（网格参数）模型结果
"""

model_result_grid = load_data_from_pickle(Data_path,'%s模型结果.pkl' % (model_label + 'XGBGrid'))
plot_obj.plot_for_the_model(Result_path, model_result_grid, with_test=True)

# 查看XGBGrid（网格参数）对应的每一颗决策树的决策点
model_result_grid['model_final'].dump_model(Result_path+'/model_result_grid_trees.txt')
# 查看XGBGrid（网格参数）每一颗决策树对应的information和population
train_model_result_grid_leafs = ss.get_xgboost_tree_leaf_dist_and_badRate(model_result_grid,X_train[selected],y_train,y_col_name)
train_model_result_grid_leafs.head(18)


"""
10.10 XGBGrid模型decile
"""
train_XGBGrid_score = model_result_grid['p_train'].apply(mt.Performance().p_to_score)
test_XGBGrid_score = model_result_grid['p_test'].apply(mt.Performance().p_to_score)

# 训练集
ks_decile_XGBGrid_train = mt.Performance().calculate_ks_by_decile(train_XGBGrid_score, np.array(y_train), 'decile', q=10)
ks_decile_XGBGrid_train

# 测试集
point_bounds_XGBGrid = mt.BinWoe().obtain_boundaries(ks_decile_XGBGrid_train[u'分箱'])['cut_boundaries']
ks_decile_XGBGrid_test = mt.Performance().calculate_ks_by_decile(test_XGBGrid_score, np.array(y_test), 'decile', manual_cut_bounds=point_bounds_XGBGrid)
ks_decile_XGBGrid_test

# 生成XGBGrid模型runbook
run_XGBGrid_book = mt.Performance().calculate_ks_by_decile(pd.concat([train_XGBGrid_score,test_XGBGrid_score],join='outer'), np.array(pd.concat([y_train,y_test,],join='outer')), 'decile', q=20)
run_XGBGrid_book


################################################################
#S11: OOT数据验证（如果没有OOT数据可以不做这一步）
################################################################
data_OOT = load_data_from_pickle(Data_path,'data_OOT1.pkl')
data_OOT_final_selected = data_OOT[selected]
X_cat_OOT = bin_obj.convert_to_category(data_OOT_final_selected, var_dict, rebin_spec_bin_adjusted_final)


"""
11.1 按月查看分布（PSI）以及逾期率
"""
score_month = 'score_mon'
data_OOT[score_month] = data_OOT.score_time.apply(lambda x : str(x)[:7])

X_cat_oot_with_y_appmon = pd.merge(X_cat_OOT,data_OOT[[y_col_name, score_month]],left_index=True,right_index=True)
# 这一步按月统计了变量的分布以及缺失率，这一步可以将字段先存储出去，观察不合格的变量，在下一步可以提前删除
var_dist_badRate_by_time_oot = ss.get_badRate_and_dist_by_time(X_cat_oot_with_y_appmon,woe_iv_df_coarse,selected_final,score_month,y_col_name)

# ## xgb数据转化
data_OOT_final_xgboost = mt.BinWoe().apply_xgboost_data_derive(data_OOT_final_selected,
                                var_dict, rebin_spec, bin_to_label, dummy_var_name_map=dummy_var_name_map)
X_OOT_xgboost_with_bin_dummy = bin_obj.convert_xgboost_rebins_to_dummy(data_OOT_final_xgboost,X_cat_OOT, dummy_var_name_map,is_apply=True)

# ### 使用XGBExp模型结果对OOT打分
# 由于建模时候apollo_osBalanceSelfProduct有-8887取值，但是OOT中没有，因此出现了dummy的缺失
X_OOT_xgboost_with_bin_dummy['recent_180D_tongdunBlackDecisionDUMMY0'] = 0
data_OOT_XGBExp_p_score = mt.Performance().calculate_score_by_xgb_model_result(X_OOT_xgboost_with_bin_dummy, model_result_exp)
data_OOT_XGBExp = pd.merge(data_OOT,data_OOT_XGBExp_p_score,left_index=True,right_index=True)

# 生成runbook
run_book_XGBExp_OOT = mt.Performance().calculate_ks_by_decile(data_OOT_XGBExp.xgbScore,data_OOT_XGBExp[y_col_name], 'decile', q=20)
run_book_XGBExp_OOT

# 生成decile
ks_decile_OOT_XGBExp = mt.Performance().calculate_ks_by_decile(data_OOT_XGBExp.xgbScore, data_OOT_XGBExp[y_col_name], 'decile',manual_cut_bounds=point_bounds_XGBExp)
ks_decile_OOT_XGBExp

# ### 使用XGBRandom模型结果对OOT打分
data_OOT_XGBRandom_p_score = mt.Performance().calculate_score_by_xgb_model_result(X_OOT_xgboost_with_bin_dummy, model_result_random)
# 将打分数据与y数据进行合并
data_OOT_XGBRandom = pd.merge(data_OOT,data_OOT_XGBRandom_p_score,left_index=True,right_index=True)

# 生成runbook
run_book_XGBRandom_OOT = mt.Performance().calculate_ks_by_decile(data_OOT_XGBRandom.xgbScore,data_OOT_XGBRandom[y_col_name], 'decile', q=20)
run_book_XGBRandom_OOT

# 生成decile
ks_decile_OOT_XGBRandom = mt.Performance().calculate_ks_by_decile(data_OOT_XGBRandom.xgbScore, data_OOT_XGBRandom[y_col_name], 'decile', manual_cut_bounds=point_bounds_XGBRandom)
ks_decile_OOT_XGBRandom

# ### 使用XGBGrid模型结果对OOT打分
data_OOT_XGBGrid_p_score = mt.Performance().calculate_score_by_xgb_model_result(X_OOT_xgboost_with_bin_dummy, model_result_grid)
# 将打分数据与y数据进行合并
# 将打分数据与y数据进行合并
data_OOT_XGBGrid = pd.merge(data_OOT,data_OOT_XGBGrid_p_score,left_index=True,right_index=True)

# 生成runbook
run_book_XGBGrid_OOT = mt.Performance().calculate_ks_by_decile(data_OOT_XGBGrid.xgbScore,data_OOT_XGBGrid[y_col_name], 'decile', q=20)
run_book_XGBGrid_OOT

# 生成decile
ks_decile_OOT_XGBGrid = mt.Performance().calculate_ks_by_decile(data_OOT_XGBGrid.xgbScore, data_OOT_XGBGrid[y_col_name], 'decile', manual_cut_bounds=point_bounds_XGBGrid)
ks_decile_OOT_XGBGrid

# ### 使用XGBHyperopt模型结果对OOT打分
data_OOT_XGBHyperopt_p_score = mt.Performance().calculate_score_by_xgb_model_result(X_OOT_xgboost_with_bin_dummy, model_result_hyperopt)
# 将打分数据与y数据进行合并
# 将打分数据与y数据进行合并
data_OOT_XGBHyperopt = pd.merge(data_OOT,data_OOT_XGBHyperopt_p_score ,left_index=True,right_index=True)

# 生成runbook
run_book_yzd_yyq_XGBHyperopt_OOT = mt.Performance().calculate_ks_by_decile(data_OOT_XGBHyperopt.xgbScore,data_OOT_XGBHyperopt[y_col_name], 'decile', q=20)
run_book_yzd_yyq_XGBHyperopt_OOT

# 生成decile
ks_decile_OOT_XGBHyperopt_yzd_yyq = mt.Performance().calculate_ks_by_decile(data_OOT_XGBHyperopt.xgbScore, data_OOT_XGBHyperopt[y_col_name], 'decile',manual_cut_bounds=point_bounds_XGBHyperopt)
ks_decile_OOT_XGBHyperopt_yzd_yyq

# # OOT PSI以及ranking验证
# 合并y数据为后续统计做准备
data_OOT_final_xgboost_with_y = pd.merge(X_OOT_xgboost_with_bin_dummy, data_OOT[[y_col_name]],left_index=True,right_index=True)

# ## XGBExp模型分的psi
score_ranking_train_vs_oot_XGBExp = mt.Performance().score_ranking_psi_train_vs_oot(ks_decile_XGBExp_train,ks_decile_OOT_XGBExp)
score_ranking_train_vs_oot_XGBExp

# ### XGBExp每颗树叶子psi以及badRate
oot_model_result_exp_leafs = ss.get_xgboost_tree_leaf_dist_and_badRate(model_result_exp, data_OOT_final_xgboost_with_y.drop(y_col_name, axis=1),data_OOT_final_xgboost_with_y[y_col_name], y_col_name)
train_oot_leafs_exp_psi_badRate = ss.compare_train_OOT_leaf_psi_and_badRate(train_model_result_exp_leafs,'train',oot_model_result_exp_leafs,'oot')
train_oot_leafs_exp_psi_badRate.head()

# ## XGBRandom模型分的psi
score_ranking_train_vs_oot_XGBRandom = mt.Performance().score_ranking_psi_train_vs_oot(ks_decile_XGBRandom_train,ks_decile_OOT_XGBRandom)
score_ranking_train_vs_oot_XGBRandom

# ###  XGBRandom每颗树叶子psi以及badRate
oot_model_result_random_leafs = ss.get_xgboost_tree_leaf_dist_and_badRate(model_result_random ,data_OOT_final_xgboost_yzd_yyq_with_y.drop(y_col_name,axis=1),data_OOT_final_xgboost_with_y[y_col_name],y_col_name)
train_oot_leafs_random_psi_badRate = ss.compare_train_OOT_leaf_psi_and_badRate(train_model_result_random_leafs,'train', oot_model_result_random_leafs,'oot')
train_oot_leafs_random_psi_badRate.head(10)

# ## XGBGrid模型分psi
score_ranking_train_vs_oot_XGBGrid = mt.Performance().score_ranking_psi_train_vs_oot(ks_decile_XGBGrid_train,ks_decile_OOT_XGBGrid)
score_ranking_train_vs_oot_XGBGrid

# ### XGBGrid每颗树叶子psi以及badRate
oot_model_result_grid_leafs = ss.get_xgboost_tree_leaf_dist_and_badRate(model_result_grid,data_OOT_final_xgboost_with_y.drop(y_col_name, axis=1),data_OOT_final_xgboost_with_y[y_col_name],y_col_name)
train_oot_leafs_grid_psi_badRate = ss.compare_train_OOT_leaf_psi_and_badRate(train_model_result_grid_leafs,'train', oot_model_result_grid_leafs,'oot')
train_oot_leafs_grid_psi_badRate.head(18)

# ## XGBHyperopt模型分psi
score_ranking_train_vs_oot_XGBHyperopt = mt.Performance().score_ranking_psi_train_vs_oot(ks_decile_XGBHyperopt_train, ks_decile_OOT_XGBHyperopt)
score_ranking_train_vs_oot_XGBHyperopt

# ### XGBHyperopt每颗树叶子psi以及badRate
oot_model_result_hyperopt_leafs = ss.get_xgboost_tree_leaf_dist_and_badRate(model_result_hyperopt, data_OOT_final_xgboost_with_y.drop(y_col_name, axis=1),data_OOT_final_xgboost_with_y[y_col_name],y_col_name)
train_oot_leafs_hyperopt_psi_badRate = ss.compare_train_OOT_leaf_psi_and_badRate(train_model_result_hyperopt_leafs,'train', oot_model_result_hyperopt_leafs,'oot')
train_oot_leafs_hyperopt_psi_badRate.head(10)

################################################################
#S12: 模型部署
################################################################
import utils3.deploy as dp 
reload(dp)


"""
12.1 保存模型分类器, 保存结果可提供给开发使用
"""

# hyperopt模型
hyperopt_model = load_data_from_pickle(Data_path, '{}XGBHyperopt模型结果.pkl'.format(model_label))
hyperopt_model['model_final'].save_model(os.path.join(Data_path, '{}_XGBHyperopt_model.pkl').format(model_label))

# exp模型
exp_model = load_data_from_pickle(Data_path, '{}XGBExp模型结果.pkl'.format(model_label))
exp_model['model_final'].save_model(os.path.join(Data_path, '{}_XGBExp_model.pkl').format(model_label))

# grid模型
grid_model = load_data_from_pickle(Data_path, '{}XGBGrid模型结果.pkl'.format(model_label))
grid_model['model_final'].save_model(os.path.join(Data_path, '{}_XGBGrid_model.pkl').format(model_label))

# random模型
random_model = load_data_from_pickle(Data_path, '{}XGBRandom模型结果.pkl'.format(model_label))
random_model['model_final'].save_model(os.path.join(Data_path, '{}_XGBRandom_model.pkl').format(model_label))

"""
12.2 生成模型文档
"""

FINAL_REPORT_PATH = Root_path
import utils3.filing as fl
data_dict = {}
data_dict['01_模型结果'] = {
    '样本统计': sample_split_summary,
    '好坏分布': '1_Good_VS_BAD.png',
    #如果是图片的话必须key值中包含'_picture'，然后value为figure_path下面的图存储路径和文件名 
    'XGBExp_TRAIN_KS_picture': 'KS/{}XGBExp_TRAIN_KS.png'.format(model_label),
    'XGBExp_TEST_KS_picture': 'KS/{}XGBExp_TEST_KS.png'.format(model_label),
    'XGBExp_CV5_KS_picture': 'KS/{}XGBExp_5FoldCV_KS.png'.format(model_label),
    'XGBGrid_TRAIN_KS_picture': 'KS/{}XGBGrid_TRAIN_KS.png'.format(model_label),
    'XGBGrid_TEST_KS_picture': 'KS/{}XGBGrid_TEST_KS.png'.format(model_label),
    'XGBGrid_CV5_KS_picture': 'KS/{}XGBGrid_5FoldCV_KS.png'.format(model_label),
    'XGBRandom_TRAIN_KS_picture': 'KS/{}XGBRandom_TRAIN_KS.png'.format(model_label),
    'XGBRandom_TEST_KS_picture': 'KS/{}XGBRandom_TEST_KS.png'.format(model_label),
    'XGBRandom_CV5_KS_picture': 'KS/{}XGBRandom_5FoldCV_KS.png'.format(model_label),  
    'XGBHyperopt_TRAIN_KS_picture': 'KS/{}XGBHyperopt_TRAIN_KS.png'.format(model_label),
    'XGBHyperopt_TEST_KS_picture': 'KS/{}XGBHyperopt_TEST_KS.png'.format(model_label),
    'XGBHyperopt_CV5_KS_picture': 'KS/{}XGBHyperopt_5FoldCV_KS.png'.format(model_label), 
}
data_dict['02_EDA'] = summary
data_dict['03_模型细分箱'] = woe_iv_df
data_dict['04_决策参考'] = {'XGBExp_train_decile':ks_decile_XGBExp_train,
                          'XGBExp_test_decile':ks_decile_XGBExp_test,
                      'XGBExp_runbook':run_XGBExp_book,
                      'XGBGrid_train_decile':ks_decile_XGBGrid_train,
                        'XGBGrid_test_decile':ks_decile_XGBGrid_test,
                      'XGBGrid_runbook':run_XGBGrid_book,
                      'XGBRandom_train_decile':ks_decile_XGBRandom_train,
                        'XGBRandom_test_decile':ks_decile_XGBRandom_test,
                      'XGBRandom_runbook':run_XGBRandom_book,
                       'XGBHyperopt_train_decile':ks_decile_XGBHyperopt_train,
                        'XGBHyperopt_test_decile':ks_decile_XGBHyperopt_test,
                      'XGBHyperopt_runbook':run_XGBHyperopt_book,
                      }
#data_dict['06_OOT统计'] = {'XGBExp_oot_runbook':ks_decile_OOT_XGBExp,
#                       'XGBExp_oot_PSI':score_ranking_train_vs_oot_XGBExp,
#                       'XGBGrid_oot_runbook':ks_decile_OOT_XGBGrid,
#                       'XGBGrid_oot_PSI':score_ranking_train_vs_oot_XGBGrid,
#                        'XGBRandom_oot_runbook':ks_decile_OOT_XGBRandom,
#                       'XGBRandom_oot_PSI':score_ranking_train_vs_oot_XGBRandom,
#                        'XGBHyperopt_oot_runbook':ks_decile_OOT_XGBHyperopt,
#                       'XGBHyperopt_oot_PSI':score_ranking_train_vs_oot_XGBHyperopt,
#                       }
#data_dict['06_OOT附录1变量按时间分布以及逾期率'] = var_dist_badRate_by_time_oot
#data_dict['06_OOT附录2变量重要性'] = {'EXP':fi_exp_psi,'Grid':fi_grid_psi,'random':fi_random_psi,'Hyper':fi_hyper_psi}
#data_dict['07_leafs_OOT统计'] = {'train_oot_leafs_exp_psi_badRate':train_oot_leafs_exp_psi_badRate,
#                       'train_oot_leafs_random_psi_badRate':train_oot_leafs_random_psi_badRate,
#                       'train_oot_leafs_grid_psi_badRate':train_oot_leafs_grid_psi_badRate,
#                        'train_oot_leafs_hyperopt_psi_badRate':train_oot_leafs_hyperopt_psi_badRate,
#                       }
# 如果sheet对应的内容是df，则从sheet的A1位置开始插入整张表格，不包含pd.DataFrame的index
#data_dict['细节1_变量按时间分布以及逾期率'] = var_dist_badRate_by_time
fl.ModelSummary2Excel(FINAL_REPORT_PATH, os.path.join(Result_path, 'figure'), '{}模型文档.xlsx'.format(model_label), data_dict).run()





"""
12.2 可以不执行的
"""
## ## 生成XGBExp模型部署文档
## 引入模型的分箱明细
#model_spec = load_data_from_pickle(Data_path,'{}_XGBoost输出数据和分箱明细.pkl'.format(model_label))
#xgb_importance_Exp = pd.read_excel(Result_path+'/{}XGBExp模型变量重要性排序.xlsx'.format(model_label))
#woe_iv_df_coarse = load_data_from_pickle(Result_path,'woe_iv_df_coarse.pkl')
#
#var_dict = pd.read_excel('/Users/pintec/Nextcloud/Data_Science/分析文件/modeling/new_demo_data/B_v3_var_dict_PINTEC.xlsx')
## shift+table查看每个参数含义
#dp.generate_xgb_deployment_documents('demoEXP', summary,model_spec,xgb_importance_Exp,
#                                      var_dict,ks_decile_XGBExp_train,model_result_exp,Result_path
#                                    ,X_train,1000,woe_iv_df_coarse
#                                      , production_name_map={'riskGrade':'rg'})
#
#fi_exp_psi = ss.get_xgb_fscore_with_monthly_PSI(model_result_exp,var_dist_badRate_by_time_oot)
#fi_exp_psi
## ## 生成XGBGrid模型部署文档
## 引入模型的分箱明细
#model_spec = load_data_from_pickle(Data_path,'%s_XGBoost输出数据和分箱明细.pkl' % 'demo_xgb')
#var_dict = pd.read_excel('/Users/pintec/Nextcloud/Data_Science/分析文件/modeling/new_demo_data/B_v3_var_dict_PINTEC.xlsx')
#xgb_importance_Grid = pd.read_excel(Result_path+'/demo_xgbXGBGrid模型变量重要性排序.xlsx')
#woe_iv_df_coarse = load_data_from_pickle(Result_path,'woe_iv_df_coarse.pkl')
#
## shift+table查看每个参数含义
#dp.generate_xgb_deployment_documents('demoGrid', summary,model_spec,xgb_importance_Grid,
#                                      var_dict,ks_decile_XGBGrid_train,model_result_grid,Result_path,
#                                     X_train,1000,woe_iv_df_coarse
#                                      , production_name_map={})
#
#fi_grid_psi = ss.get_xgb_fscore_with_monthly_PSI(model_result_grid,var_dist_badRate_by_time_oot)
## ## 生成XGBRandom模型部署文档
## 引入模型的分箱明细
#model_spec = load_data_from_pickle(Data_path,'%s_XGBoost输出数据和分箱明细.pkl' % 'demo_xgb')
#var_dict = pd.read_excel('/Users/pintec/Nextcloud/Data_Science/分析文件/modeling/new_demo_data/B_v3_var_dict_PINTEC.xlsx')
#xgb_importance_Random = pd.read_excel(Result_path+'/demo_xgbXGBRandom模型变量重要性排序.xlsx')
#woe_iv_df_coarse = load_data_from_pickle(Result_path,'woe_iv_df_coarse.pkl')
## shift+table查看每个参数含义
#dp.generate_xgb_deployment_documents('demoRandom', summary,model_spec,xgb_importance_Random,
#                                      var_dict,ks_decile_XGBRandom_train,model_result_random,Result_path
#                                     ,X_train,1000,woe_iv_df_coarse 
#                                     , production_name_map={'riskGrade':'RG'})
#fi_random_psi = ss.get_xgb_fscore_with_monthly_PSI(model_result_random,var_dist_badRate_by_time_oot)
## ## 生成XGBHyperopt模型部署文档
## 引入模型的分箱明细
#model_spec = load_data_from_pickle(Data_path,'%s_XGBoost输出数据和分箱明细.pkl' % 'demo_xgb')
#var_dict = pd.read_excel('/Users/pintec/Nextcloud/Data_Science/分析文件/modeling/new_demo_data/B_v3_var_dict_PINTEC.xlsx')
#xgb_importance_Hyperopt = pd.read_excel(Result_path+'/demo_xgbXGBHyperopt模型变量重要性排序.xlsx')
#woe_iv_df_coarse = load_data_from_pickle(Result_path,'woe_iv_df_coarse.pkl')
## shift+table查看每个参数含义
#dp.generate_xgb_deployment_documents('demoHyperopt', summary, model_spec,xgb_importance_Hyperopt,
#                                      var_dict,ks_decile_XGBHyperopt_train,model_result_hyperopt,Result_path
#                                      ,X_train,1000,woe_iv_df_coarse,
#                                     production_name_map={'riskGrade':'RG'})
#
#fi_hyper_psi = ss.get_xgb_fscore_with_monthly_PSI(model_result_hyperopt,var_dist_badRate_by_time_oot)
