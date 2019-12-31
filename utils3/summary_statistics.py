# encoding=utf8
"""
对数据进行Exploratory data analysis和极值处理

Owner： 胡湘雨

最近更新：2017-12-07
"""
import os

import pandas as pd
import numpy as np
from jinja2 import Template

try:
    import xgboost as xgb
except:
    pass

from utils3.data_io_utils import *
from utils3.misc_utils import *
import utils3.metrics as mt
import utils3.misc_utils as mu
from math import log


# useless_vars = [u'n_mobileNo',  u'c_idcardno_encrypt', u'c_address', \
# u'c_custName', u'c_ecpCellphone_encrypt',u'c_mobile_encrypt', u'c_idcardno',\
# u'n_idcardBirthDate', u'c_userCookieID', u'n_userCookieID', \
# u'c_ecpRealName', u'c_loanContract',u'c_userIP', u'c_tokenId', u'c_userGroup',\
# u'c_Comments', u'c_GameGroup', u'c_blackBox', u'n_modelScore', u'n_henglinApiStatus',\
# u'n_unionpayCardNo', u'n_finalScore', u'n_consultingFeeRate', u'n_isSmallLoan',\
# u'n_matePhonenumber', u'n_fraudVal', u'n_fraudVal2', u'n_SpecificRiskScore'
# ]

# exempt_vars = [u'c_client', u'c_province', u'n_age', u'n_currentJobyear', \
# u'n_cardTerm', u'n_goOut120', u'n_avgMonthCall', u'n_call110', u'n_longTimeShutdown',\
# u'n_networkTime6', u'n_zhimaScore', u'n_ivsScore']


# add by jzw for eda sample_weight
def eda_sw(X,var_dict,sample_weight,useless_vars,exempt_vars,data_path,save_label,uniqvalue_cutoff=0.97):
    eda_sample_weight=[]
    for col in X.columns.values:
        #print(col)
        tmp={}
        sample_weight=pd.DataFrame(sample_weight)
        mid_data=X[col].to_frame(col).merge(sample_weight,left_index=True,right_index=True,how='inner')
        tmp[u'指标英文']=col
        tmp['N']=mid_data['sample_weight'].sum()
        tmp['N_-9999']=mid_data[mid_data[col].isin([-9999,'-9999','-9999.0'])]['sample_weight'].sum()
        tmp['pct_-9999']=np.round(tmp['N_-9999']/tmp['N'],3)
        tmp['N_-8888']=mid_data[mid_data[col].isin([-8888,'-8888','-8888.0'])]['sample_weight'].sum()
        tmp['pct_-8888']=np.round(tmp['N_-8888']/tmp['N'],3)
        tmp['N_-8887']=mid_data[mid_data[col].isin([-8887,'-8887','-8887.0'])]['sample_weight'].sum()
        tmp['pct_-8887']=np.round(tmp['N_-8887']/tmp['N'],3)
        tmp['pct_NA']=tmp['pct_-9999']+tmp['pct_-8888']+tmp['pct_-8887']
        tmp['N_0']=mid_data[mid_data[col]==0]['sample_weight'].sum()
        tmp['pct_0']=np.round(tmp['N_0']/tmp['N'],3)
        if var_dict[var_dict[u'指标英文']==col]['数据类型'].tolist()[0]=='varchar':
            tmp['N_categories']=mid_data[col].nunique()
            mid_data=mid_data.replace('-8888.0',np.nan).replace('-9999.0', np.nan).replace('-8887.0',np.nan)\
            .replace('-8888',np.nan).replace('-8887',np.nan).replace('-9999',np.nan)
            tmp_data=pd.pivot_table(mid_data,index=col,values='sample_weight',aggfunc=np.sum).reset_index()
            tmp_data.sort_values(['sample_weight'],ascending=False,inplace=True)
            tmp_data.reset_index(drop=True,inplace=True)
            if tmp_data.shape[0]>2:
                tmp['1st类别']=str(tmp_data[col][0])+'#='+str(tmp_data['sample_weight'][0])+',%='+str(tmp_data['sample_weight'][0]/tmp_data['sample_weight'].sum())
                tmp['2nd类别']=str(tmp_data[col][1])+'#='+str(tmp_data['sample_weight'][1])+',%='+str(tmp_data['sample_weight'][1]/tmp_data['sample_weight'].sum())
                tmp['3rd类别']=str(tmp_data[col][2])+'#='+str(tmp_data['sample_weight'][2])+',%='+str(tmp_data['sample_weight'][2]/tmp_data['sample_weight'].sum())
            elif tmp_data.shape[0]==2:
                tmp['1st类别']=str(tmp_data[col][0])+'#='+str(tmp_data['sample_weight'][0])+',%='+str(tmp_data['sample_weight'][0]/tmp_data['sample_weight'].sum())
                tmp['2nd类别']=str(tmp_data[col][1])+'#='+str(tmp_data['sample_weight'][1])+',%='+str(tmp_data['sample_weight'][1]/tmp_data['sample_weight'].sum())
            elif tmp_data.shape[0]==1:
                tmp['1st类别']=str(tmp_data[col][0])+'#='+str(tmp_data['sample_weight'][0])+',%='+str(tmp_data['sample_weight'][0]/tmp_data['sample_weight'].sum())
            else:
                pass
            if tmp['N_categories']==1:
                tmp['exclusion_reason']='只有一个分类'
            elif tmp['N_categories']>100:
                tmp['exclusion_reason']='类别变量的类别数过多'
            else:
                tmp['exclusion_reason']=None
        else:
            mid_data=mid_data.replace(-8888,np.nan).replace(-8887,np.nan).replace(-9999,np.nan)
            tmp_data=pd.pivot_table(mid_data,index=col,values='sample_weight',aggfunc=np.sum).reset_index()
            tmp_data.sort_values(col,ascending=False,inplace=True)
            tmp_data['pct_sw']=tmp_data['sample_weight']/tmp_data['sample_weight'].sum()
            tmp_data['pct']=round(tmp_data['pct_sw'].cumsum(),2)
            tmp['mean']=(tmp_data[col]*tmp_data['sample_weight']).sum()/tmp_data['sample_weight'].sum()
            tmp['std']=pow((tmp_data[col]-tmp['mean']),2).sum()/(tmp_data['sample_weight'].sum()-1)
            min_pct=min(tmp_data['pct'])
            if min_pct<=0.5:
                tmp['median']=max(tmp_data.loc[tmp_data['pct']<=0.5,col])
            else:
                tmp['median']=max(tmp_data.loc[tmp_data['pct']<=min_pct,col])
            tmp['min']=min(mid_data[col])
            tmp['max']=max(mid_data[col])
            if min_pct<=0.01:
                tmp['p01']=max(tmp_data.loc[tmp_data['pct']<=0.01,col])
            else:
                tmp['p01']=max(tmp_data.loc[tmp_data['pct']<=min_pct,col])
            if min_pct<=0.05:
                tmp['p05']=max(tmp_data.loc[tmp_data['pct']<=0.05,col])
            else:
                tmp['p05']=max(tmp_data.loc[tmp_data['pct']<=min_pct,col])
            if min_pct<=0.1:
                tmp['p10']=max(tmp_data.loc[tmp_data['pct']<=0.1,col])
            else:
                tmp['p10']=max(tmp_data.loc[tmp_data['pct']<=min_pct,col])
            if min_pct<=0.25:
                tmp['p25']=max(tmp_data.loc[tmp_data['pct']<=0.25,col])
            else:
                tmp['p25']=max(tmp_data.loc[tmp_data['pct']<=min_pct,col])
            if min_pct<=0.75:
                tmp['p75']=max(tmp_data.loc[tmp_data['pct']<=0.75,col])
            else:
                tmp['p75']=max(tmp_data.loc[tmp_data['pct']<=min_pct,col])
            if min_pct<=0.9:
                tmp['p90']=max(tmp_data.loc[tmp_data['pct']<=0.9,col])
            else:
                tmp['p90']=max(tmp_data.loc[tmp_data['pct']<=min_pct,col])
            if min_pct<=0.95:
                tmp['p95']=max(tmp_data.loc[tmp_data['pct']<=0.95,col])
            else:
                tmp['p95']=max(tmp_data.loc[tmp_data['pct']<=min_pct,col])
            if min_pct<=0.99:
                tmp['p99']=max(tmp_data.loc[tmp_data['pct']<=0.99,col])
            else:
                tmp['p99']=max(tmp_data.loc[tmp_data['pct']<=min_pct,col])
        if tmp['pct_NA']>uniqvalue_cutoff:
            tmp['exclusion_reason']='缺失NA比例大于{}'.format(uniqvalue_cutoff)
        elif tmp['pct_0']>uniqvalue_cutoff:
            tmp['exclusion_reason']='0值比例大于{}'.format(uniqvalue_cutoff)
        else:
            tmp['exclusion_reason']=None
        if col in useless_vars:
            tmp['exclusion_reason']='无用变量'
        elif col in exempt_vars:
            tmp['exclusion_reason']=None
        else:
            tmp['exclusion_reason']=tmp['exclusion_reason']
        eda_sample_weight.append(tmp)
    eda_sample_weight=pd.DataFrame(eda_sample_weight)
    eda=pd.merge(eda_sample_weight,var_dict,on=u'指标英文',how='inner')
    if 'N_categories' in eda.columns:
        eda['N_categories'].fillna(0,inplace=True)
    else:
        pass
    if 'N_categories' in eda.columns and max(eda['N_categories'])>=3:
        eda=eda[[u'数据源',u'指标英文',u'指标中文',u'数据类型',u'指标类型','exclusion_reason','N','pct_NA','N_-9999','pct_-9999',\
             'N_-8888','pct_-8888','N_-8887','pct_-8887','N_0','pct_0','mean','std','median','min','max','p01','p05',\
             'p10','p25','p75','p90','p95','p99','N_categories','1st类别','2nd类别','3rd类别']]
    elif 'N_categories' in eda.columns and max(eda['N_categories'])==2:
        eda=eda[[u'数据源',u'指标英文',u'指标中文',u'数据类型',u'指标类型','exclusion_reason','N','pct_NA','N_-9999','pct_-9999',\
             'N_-8888','pct_-8888','N_-8887','pct_-8887','N_0','pct_0','mean','std','median','min','max','p01','p05',\
             'p10','p25','p75','p90','p95','p99','N_categories','1st类别','2nd类别']]
    elif 'N_categories' in eda.columns and max(eda['N_categories'])==1:
        eda=eda[[u'数据源',u'指标英文',u'指标中文',u'数据类型',u'指标类型','exclusion_reason','N','pct_NA','N_-9999','pct_-9999',\
             'N_-8888','pct_-8888','N_-8887','pct_-8887','N_0','pct_0','mean','std','median','min','max','p01','p05',\
             'p10','p25','p75','p90','p95','p99','N_categories','1st类别']]
    else:
        eda=eda[[u'数据源',u'指标英文',u'指标中文',u'数据类型',u'指标类型','exclusion_reason','N','pct_NA','N_-9999','pct_-9999',\
             'N_-8888','pct_-8888','N_-8887','pct_-8887','N_0','pct_0','mean','std','median','min','max','p01','p05',\
             'p10','p25','p75','p90','p95','p99']]
    eda.to_excel(data_path+'/%s_variables_summary.xlsx'% save_label)
    return


def eda(X, var_dict, useless_vars, exempt_vars, data_path, save_label, uniqvalue_cutoff=0.97):
    """
    计算N, missing value percent, mean, min, max, percentiles 等。

    Args:

    X (pd.DataFrame()): 是一个宽表，每一列是一个变量，每一行是一个obs
    var_dict (pd.DataFrame()): 标准变量字典表，包含以下这些列：数据源，数据类型，指标类型，
    指标英文，指标中文，变量解释，取值解释。
    useless_vars (list): 无用变量名list
    exempt_vars (list): 豁免变量名list，这些变量即使一开始被既定原因定为exclude，也会被
        保留，比如前一版本模型的变量
    data_path (str): 存储数据文件路径
    save_label (str): summary文档存储将用'%s_variables_summary.xlsx' % save_label存
    uniqvalue_cutoff（float): 0-1之间，缺失率和唯一值阈值设定

    Returns：
    None
    直接将结果存成xlsx
    """
    variable_summary = X.count(0).to_frame('N_available')
    variable_summary.loc[:, 'N'] = len(X)
    variable_summary.loc[:, 'N_-9999'] = (X.isin([-9999, '-9999', '-9999.0'])).sum()
    variable_summary.loc[:, 'pct_-9999'] = np.round(variable_summary['N_-9999'] * 1.0 / variable_summary.N, 3)
    variable_summary.loc[:, 'N_-8888'] = (X.isin([-8888, '-8888', '-8888.0'])).sum()
    variable_summary.loc[:, 'pct_-8888'] = np.round(variable_summary['N_-8888'] * 1.0 / variable_summary.N, 3)
    variable_summary.loc[:, 'N_-8887'] = (X.isin([-8887, '-8887', '-8887.0'])).sum()
    variable_summary.loc[:, 'pct_-8887'] = np.round(variable_summary['N_-8887'] * 1.0 / variable_summary.N, 3)
    variable_summary.loc[:, 'pct_NA'] = variable_summary['pct_-9999'] \
                                        + variable_summary['pct_-8888'] \
                                        + variable_summary['pct_-8887']
    variable_summary.loc[:, 'N_0'] = (X == 0).sum()
    variable_summary.loc[:, 'pct_0'] = np.round(variable_summary.N_0 * 1.0 / variable_summary.N, 3)
    # the following can only be done for continuous variable
    try:
        numerical_vars = var_dict.loc[(var_dict['数据类型'] != 'varchar') & ~pd.isnull(var_dict['指标英文']), '指标英文'].tolist()
        categorical_vars = var_dict.loc[(var_dict['数据类型'] == 'varchar') & ~pd.isnull(var_dict['指标英文']), '指标英文'].tolist()
        numerical_vars = list(set(numerical_vars).intersection(set(X.columns)))
        categorical_vars = list(set(categorical_vars).intersection(set(X.columns)))
    except:
        numerical_vars = X.dtypes[X.dtypes!='object'].index.values
        categorical_vars = X.dtypes[X.dtypes=='object'].index.values

    if len(numerical_vars) > 0:
        X_numerical = X[numerical_vars].apply(lambda x: x.astype(float), 0)\
                                       .replace(-8888, np.nan)\
                                       .replace(-9999, np.nan)\
                                       .replace(-8887, np.nan)
        numerical_vars_summary = X_numerical.mean().round(1).to_frame('mean')
        numerical_vars_summary.loc[:, 'std'] = X_numerical.std().round(1)
        numerical_vars_summary.loc[:, 'median'] = X_numerical.median().round(1)
        numerical_vars_summary.loc[:, 'min'] = X_numerical.min()
        numerical_vars_summary.loc[:, 'max'] = X_numerical.max()
        numerical_vars_summary.loc[:, 'p01'] = X_numerical.quantile(0.01)
        numerical_vars_summary.loc[:, 'p05'] = X_numerical.quantile(q=0.05)
        numerical_vars_summary.loc[:, 'p10'] = X_numerical.quantile(q=0.10)
        numerical_vars_summary.loc[:, 'p25'] = X_numerical.quantile(q=0.25)
        numerical_vars_summary.loc[:, 'p75'] = X_numerical.quantile(q=0.75)
        numerical_vars_summary.loc[:, 'p90'] = X_numerical.quantile(q=0.90)
        numerical_vars_summary.loc[:, 'p95'] = X_numerical.quantile(q=0.95)
        numerical_vars_summary.loc[:, 'p99'] = X_numerical.quantile(q=0.99)
    # the following are for categorical_vars
    if len(categorical_vars) > 0:
        X_categorical = X[categorical_vars].copy()
        X_categorical = X_categorical.replace('-8888.0', np.nan)\
                                     .replace('-9999.0', np.nan)\
                                     .replace('-8887.0', np.nan)\
                                     .replace('-8888', np.nan)\
                                     .replace('-9999', np.nan)\
                                     .replace('-8887', np.nan)
        categorical_vars_summary = X_categorical.nunique().to_frame('N_categories')
        cat_list = []
        for col in categorical_vars:
            if X_categorical[col].count() == 0:
                pass
            else:
                cat_count = X_categorical[col].value_counts().head(3)
                if len(cat_count) == 3:
                    col_result = pd.Series({'1st类别': str(cat_count.index.values[0]) + ' #=' + str(cat_count.iloc[0])\
                                                     + ', %=' + str(np.round(cat_count.iloc[0] * 1.0 / len(X), 2)),\
                                            '2nd类别': str(cat_count.index.values[1]) + ' #=' + str(cat_count.iloc[1])\
                                                     + ', %=' + str(np.round(cat_count.iloc[1] * 1.0 / len(X), 2)),\
                                            '3rd类别': str(cat_count.index.values[2]) + ' #=' + str(cat_count.iloc[2])\
                                                     + ', %=' + str(np.round(cat_count.iloc[2] * 1.0 / len(X), 2))\
                                            })\
                                            .to_frame().transpose()
                elif len(cat_count) == 2:
                    col_result = pd.Series({'1st类别': str(cat_count.index.values[0]) + ' #=' + str(cat_count.iloc[0])\
                                                     + ', %=' + str(np.round(cat_count.iloc[0] * 1.0 / len(X), 2)),\
                                            '2nd类别': str(cat_count.index.values[1]) + ' #=' + str(cat_count.iloc[1])\
                                                     + ', %=' + str(np.round(cat_count.iloc[1] * 1.0 / len(X), 2)),\
                                            })\
                                            .to_frame().transpose()
                elif len(cat_count) == 1:
                    col_result = pd.Series({'1st类别': str(cat_count.index.values[0]) + ' #=' + str(cat_count.iloc[0])\
                                                    + ', %=' + str(np.round(cat_count.iloc[0] * 1.0 / len(X), 2))})\
                                            .to_frame().transpose()
                else:
                    pass

                col_result.index = [col]
                cat_list.append(col_result)

        cat_df = pd.concat(cat_list)

    # merge all summaries
    if len(numerical_vars) > 0 and len(categorical_vars) > 0:
        all_variable_summary = variable_summary.merge(numerical_vars_summary, how='left', left_index=True, right_index=True)
        all_variable_summary = all_variable_summary.merge(categorical_vars_summary, how='left', left_index=True, right_index=True)
        all_variable_summary = all_variable_summary.merge(cat_df, how='left', left_index=True, right_index=True)
    elif len(numerical_vars) > 0 and len(categorical_vars) == 0:
        all_variable_summary = variable_summary.merge(numerical_vars_summary, how='left', left_index=True, right_index=True)
    elif len(categorical_vars) > 0 and len(numerical_vars) == 0:
        all_variable_summary = variable_summary.merge(categorical_vars_summary, how='left', left_index=True, right_index=True)
        all_variable_summary = all_variable_summary.merge(cat_df, how='left', left_index=True, right_index=True)
    else:
        return None

    all_variable_summary.loc[:, 'exclusion_reason'] = None
    all_variable_summary.loc[all_variable_summary.pct_NA > uniqvalue_cutoff, 'exclusion_reason'] = '缺失NA比例大于{}'.format(uniqvalue_cutoff)
    all_variable_summary.loc[all_variable_summary.pct_0 > uniqvalue_cutoff, 'exclusion_reason'] = '0值比例大于0.97'.format(uniqvalue_cutoff)
    if len(categorical_vars) > 0:
        all_variable_summary.loc[all_variable_summary.N_categories == 1, 'exclusion_reason'] = '只有一个分类'
        all_variable_summary.loc[all_variable_summary.N_categories > 100, 'exclusion_reason'] = '类别变量的类别数过多'
    all_variable_summary.loc[useless_vars, 'exclusion_reason'] = '无用变量'

    vincio_vars = [i for i in all_variable_summary.index.values if isinstance(i, str) and 'vincio' in i]
    vincio_duplicate = [i.replace('vincio_', '') for i in vincio_vars]
    vincio_duplicate = list(set(vincio_duplicate).intersection(set(all_variable_summary.index.values)))
    vincio_vars = vincio_vars + vincio_duplicate
    all_variable_summary.loc[vincio_vars, 'exclusion_reason'] = 'vincio变量'

    bairong_vars = [i for i in all_variable_summary.index.values if isinstance(i, str) and 'al_m6' in i]
    all_variable_summary.loc[bairong_vars, 'exclusion_reason'] = 'bairong变量'

    # change exempt_vars exclusion_reason to None
    all_variable_summary.loc[exempt_vars, 'exclusion_reason'] = None


    # check
    # all_variable_summary.exclusion_reason.value_counts()
    # save
    all_variable_summary = all_variable_summary.drop('N_available', 1)
    final_output = all_variable_summary.reset_index()\
                             .rename(columns={'index':'var_code',\
                                              'pct_NA': 'NA占比',\
                                              'N_0': '0值数量',\
                                              'pct_0': '0值占比',\
                                              'N_-9999': '-9999值数量',\
                                              'pct_-9999': '-9999值占比',\
                                              'N_-8888': '-8888值数量',\
                                              'pct_-8888': '-8888值占比',\
                                              'N_-8887': '-8887值数量',\
                                              'pct_-8887': '-8887值占比',\
                                              'N_categories': '类别数量'})

    # read var_code explanation
    final_output = var_dict.loc[:, ['数据源', '指标英文', '指标中文', '数据类型', '指标类型']]\
                    .merge(final_output, left_on='指标英文', right_on='var_code')\
                    .drop('var_code', 1)

    cols = list(final_output.columns.values)
    reorder_cols = cols[:5] + [cols[-1]] + [cols[5]] + [cols[12]] + cols[6:12] +  cols[13:-1]
    if len(set(reorder_cols).intersection(set(final_output.columns))) != len(set(final_output.columns)):
        return final_output.columns
    # reorder_cols = reduce(lambda x, y: x.append(y), order_list)

    final_output = final_output[reorder_cols]
    final_output.to_excel(os.path.join(data_path, '%s_variables_summary.xlsx' % save_label), \
                        encoding='utf-8', index=False)


def cap_extreme_1(x):
    """
    将1个numerical变量的最小值cap在1th percentile,最大值cap在99th percentile
    """
    x = x.astype(float).copy()
    p01 = x.quantile(0.01)
    p99 = x.quantile(0.99)
    x.loc[x < p01] = p01
    x.loc[x > p99] = p99
    return x

def process_extreme_values(X):
    """
    将numerical变量的最小值cap在1th percentile,最大值cap在99th percentile

    Args:
    X (pd.DataFrame()): 是一个宽表，每一列是一个变量，每一行是一个obs

    Returns:
    X (pd.DataFrame()): 是一个宽表，每一列是一个变量，每一行是一个obs, 已处理好极值
    """
    numerical_vars = [i for i in X.columns.values if 'n_' in i]

    X.loc[:, numerical_vars] = X[numerical_vars].apply(cap_extreme_1, 0)
    return X

def calculate_num(label,data,sum_num):
    '''
    该函数计算含sample_weight样本的分布

    Args:

    label(string):数据标签
    data(DataFrame):['y','sample_weight']
    sum_num(float):总样本的个数

    Returns：
    tmp_list（list):
    '''
    tmp_list=[]
    tmp_list.append(label)
    total=data['sample_weight'].sum()
    tmp_list.append(total)
    good=data[data['y']==0]['sample_weight'].sum()
    tmp_list.append(good)
    bad=data[data['y']==1]['sample_weight'].sum()
    tmp_list.append(bad)
    rate=round(bad/total,4)
    tmp_list.append(rate)
    dist=round(total/sum_num,2)
    tmp_list.append(dist)

    return tmp_list


def sample_split_summary(y_train,y_test):
    '''
    该函数为训练集和测试集划分的统计结果

    Args:

    y_train(DataFrame):训练集y数据
    y_test(DataFrame):测试集的y数据

    Returns：
    New DataFrame:训练集和测试集的划分结果
    '''
    train = ['train',len(y_train),y_train.value_counts()[0]\
             ,y_train.value_counts()[1],1.0*y_train.value_counts()[1]/len(y_train),\
           1.0*len(y_train)/(len(y_train)+len(y_test))]
    test = ['test',len(y_test),y_test.value_counts()[0]\
             ,y_test.value_counts()[1],1.0*y_test.value_counts()[1]/len(y_test),\
            1.0*len(y_test)/(len(y_train)+len(y_test))]
    sum_ = ['sum',len(y_train)+len(y_test),y_train.value_counts()[0]+y_test.value_counts()[0]\
           ,y_train.value_counts()[1]+y_test.value_counts()[1],1.0*(y_train.value_counts()[1]+y_test.value_counts()[1])/(len(y_train)+len(y_test))\
           ,1.0*len(y_train)/(len(y_train)+len(y_test))+1.0*len(y_test)/(len(y_train)+len(y_test))]
    return pd.DataFrame([train,test,sum_],columns=['split','total','good','bad','rate','dist'])

def sample_split_summary_sw(y_train,y_test,sample_weight):
    '''
    该函数为训练集和测试集划分的统计结果

    Args:

    y_train(DataFrame):训练集y数据
    y_test(DataFrame):测试集的y数据
    sample_weight(DataFrame):['sample_weight'],index为applyid

    Returns：
    New DataFrame:训练集和测试集的划分结果
    '''
    #sample_weight=pd.DataFrame(sample_weight)
    sample_weight=pd.DataFrame(sample_weight)
    y_train_sw=y_train.to_frame('y').merge(sample_weight,left_index=True,right_index=True,how='inner')
    y_test_sw=y_test.to_frame('y').merge(sample_weight,left_index=True,right_index=True,how='inner')
    y_sw=pd.concat([y_train_sw,y_test_sw])
    num_sum=y_sw['sample_weight'].sum()
    y_train_list=calculate_num('train',y_train_sw,num_sum)
    y_test_list=calculate_num('test',y_test_sw,num_sum)
    y_sw_list=calculate_num('sum',y_sw,num_sum)
    data_fianl=pd.DataFrame([y_train_list,y_test_list,y_sw_list],columns=['split','total','good','bad','rate','dist'])

    return data_fianl


def get_badRate_and_dist_by_time(cat_data_with_y_and_time,woe_iv_df_coarse,select_vars,time_var_name,y_name):
    '''
    该函数为统计变量按统计时间每个分箱的逾期率以及分布以及对应的PSI

    Args:

    cat_data_with_y_and_time(DataFrame):含有时间以及y数据的data
    select_vars(list):要统计的字段
    time_var_name(varchar)：时间相关的字段名
    y_name(varchar):y的名称
    woe_iv_df_coarse(df):粗分箱文件

    Returns：
    New DataFrame:变量按统计时间每个分箱的逾期率以及分布以及对应PSI
    '''
    varchar_list = set(woe_iv_df_coarse[woe_iv_df_coarse['数据类型'] == 'varchar']['指标英文'].unique())

    varchar_in_model = varchar_list.intersection(cat_data_with_y_and_time.columns)
    if len(varchar_in_model)>0:
        for i in varchar_in_model:
            cat_data_with_y_and_time[i] = cat_data_with_y_and_time[i].astype(str)

    group_data_list = []
    for i in select_vars:
        tmp = cat_data_with_y_and_time.groupby([i,time_var_name])[y_name]\
        .agg([('allNum',len),('badNum',sum)\
          ,('badRate',lambda x : sum(x)/len(x))]).unstack()
        tmp2 = tmp.reset_index().rename(columns = {i:'bins'})
        tmp2['varName'] = i
        columns = pd.MultiIndex.from_arrays([['dist' for i in range(len(list((tmp2['allNum']/tmp2['allNum'].sum()).columns)))],
                                    list((tmp2['allNum']/tmp2['allNum'].sum()).columns)]\
                                    , names=[time_var_name,'bins'])
        tmp3 = pd.DataFrame((tmp2['allNum']/tmp2['allNum'].sum()).values, columns=columns)
        group_data_list.append(pd.concat([tmp2,tmp3],axis=1)[['varName','bins','allNum','badNum','dist','badRate']])
    
    var_dist_badRate_by_time = pd.concat(group_data_list)
    var_dist_badRate_by_time['allNum_sum'] = var_dist_badRate_by_time['allNum'].sum(axis=1)
    var_dist_badRate_by_time['badNum_sum'] = var_dist_badRate_by_time['badNum'].sum(axis=1)
    sample_size = var_dist_badRate_by_time.groupby('varName')['allNum_sum'].sum()[0]
    var_dist_badRate_by_time['dist_allNum'] = var_dist_badRate_by_time['allNum_sum']/sample_size
    var_dist_badRate_by_time['badRate_allNum'] = var_dist_badRate_by_time['badNum_sum']/var_dist_badRate_by_time['allNum_sum']

    var_dist_badRate_by_time_woe_df = pd.merge(woe_iv_df_coarse,var_dist_badRate_by_time,left_on=['指标英文','分箱']\
                                           ,right_on=['varName','bins'])

    origin_cols = var_dist_badRate_by_time_woe_df[['数据源','指标英文','指标中文','数据类型','分箱','N','坏样本数量','分布占比','逾期率','IV','KS','WOE']]
    
    columns_new = var_dist_badRate_by_time_woe_df.columns
    all_num_vars = [i for i in columns_new if i[0].startswith('allNum')]
    all_num_df = var_dist_badRate_by_time_woe_df[all_num_vars]
    bad_num_vars = [i for i in columns_new if i[0].startswith('badNum')]
    bad_num_df = var_dist_badRate_by_time_woe_df[bad_num_vars]
    
    dist_num_vars = [i for i in columns_new if i[0].startswith('dist')]
    dist_num_df = var_dist_badRate_by_time_woe_df[dist_num_vars]

    badRate_num_vars = [i for i in columns_new if i[0].startswith('badRate')]
    badRate_num_df = var_dist_badRate_by_time_woe_df[badRate_num_vars]

    var_dist_badRate_by_time_woe_df_final = pd.concat([origin_cols,all_num_df,bad_num_df,dist_num_df,badRate_num_df],axis=1)
    for i in dist_num_df:
        var_dist_badRate_by_time_woe_df_final[str(i)+'_psi'] = var_dist_badRate_by_time_woe_df_final\
        .apply(lambda x : ((x[i] - x['分布占比'])*log(x[i]/x['分布占比']) if x[i] and x['分布占比'] else 0),axis=1)

    var_dist_psi = var_dist_badRate_by_time_woe_df_final.groupby('指标英文')\
        [var_dist_badRate_by_time_woe_df_final.columns[-len(dist_num_vars):]].sum()

    var_dist_psi.columns = [str(i)+'_sum' for i in var_dist_psi.columns]
    var_dist_psi.reset_index(inplace=True)
    var_dist_badRate_by_time_woe_df_return =  pd.merge(var_dist_badRate_by_time_woe_df_final,var_dist_psi,on='指标英文')

    var_dist_badRate_by_time_woe_df_return.columns = [str(i) for i in var_dist_badRate_by_time_woe_df_return.columns]

    month_and_sum = int((len(var_dist_badRate_by_time_woe_df_return.columns) - 12)/6)

    months= [i.strip('()').split(', ')[1] for i in var_dist_badRate_by_time_woe_df_return.columns\
            if len(i.strip('()').split(', '))>1][:month_and_sum-1]+['sum']

    col2 = list(var_dist_badRate_by_time_woe_df_return.columns[:12]) + 6*months
    col1 = 12*['basic']+month_and_sum*['allNum']+month_and_sum*['badNum']\
        +month_and_sum*['dist']+month_and_sum*['badRate']\
       +month_and_sum*['bin_PSI']+month_and_sum*['sum_PSI']

    return pd.DataFrame(var_dist_badRate_by_time_woe_df_return.values,\
        columns=pd.MultiIndex.from_arrays([col1,col2]))



def get_badRate_and_dist_by_time_sw(cat_data_with_y_and_time,sample_weight,select_vars,time_var_name,y_name,var_dict):
    '''
    该函数为统计变量按统计时间每个分箱的逾期率以及分布

    Args:

    cat_data_with_y_and_time(DataFrame):含有时间以及y数据的data
    select_vars(list):要统计的字段
    time_var_name(varchar)：时间相关的字段名
    y_name(varchar):y的名称
    sample_weight:['sample_weight'],index为applyid
    var_dict:数据字典

    Returns：
    New DataFrame:变量按统计时间每个分箱的逾期率以及分布
    '''

    #data_merge=pd.merge(cat_data_with_y_and_time,sample_weight,left_index=True,right_index=True,how='inner')
    data_merge=sample_weight.to_frame('sample_weight').merge(cat_data_with_y_and_time,left_index=True,right_index=True,how='inner')
    list_fianl=[]
    for col in select_vars:
        #print(col)
        mid_time_pivot=data_merge.pivot_table(index=col,columns=[time_var_name,y_name],values='sample_weight',aggfunc=np.sum)
        time_pivot=pd.DataFrame()
        list_columns=list(mid_time_pivot.columns.values)
        #list_fianl=[]
        list_tmp=[]
        for a in list_columns:
            list_tmp.append(a[0])
        for i in list_tmp:
            mid_data=mid_time_pivot[i]
            time_pivot[i+'_eventrate']=mid_data[1]/(mid_data[0]+mid_data[1])
            time_pivot[i+'_num']=mid_data[0]+mid_data[1]
            time_pivot[i+'_dist']=time_pivot[i+'_num']/time_pivot[i+'_num'].sum()
        time_pivot_fianl=time_pivot.reset_index().rename(columns={col:'分箱'})
        time_pivot_fianl[u'指标英文']=col
        list_fianl.append(time_pivot_fianl)
    data=pd.concat(list_fianl)
    data=data.merge(var_dict[['数据源', '指标英文', '指标中文']],on=u'指标英文')
    header_columns = ['数据源', '指标英文', '指标中文', '分箱']
    other_columns = [i for i in data.columns if i not in header_columns]
    columns_order = header_columns + other_columns
    return data[columns_order]


def verify_var_and_score_validity(data_origin,score_card_deploy,rebin_spec_bin_adjusted_deploy,model_decile_deploy):
    '''
    该函数为上线以后统计是否存在缺失值未填充、打分错误以及PSI相关函数

    Args:

    data_origin(DataFrame):其中表头必须含有apply_id,var_code,create_at字段,
    大家严格按照部署监控时刻的要求撰写PSI监控时刻的SQL取数则不会出错，如果唯一标识不是apply_id，务必将唯一标识字段名修改为apply_id
    score_card_deploy(DataFrame):切记这是部署时刻的评分卡
    rebin_spec_bin_adjusted_deploy(DataFrame)：部署时刻的变量分箱.json文件
    model_decile_deploy(DataFrame):部署时刻的model decile就在部署的评分卡中

    Returns：
    wrong_data(DataFrame):未填充缺失值的所有apply
    wrong_score_vars(dict):变量以及其对应的打分错误明细
    wrong_model_score(DataFrame):模型分打分错误明细
    var_dist_data_with_score_card(DataFrame):变量PSI
    score_dist_with_decile(DataFrame):模型分PSI
    '''

    # 将部署的评分卡切换成和建模时对应的字段名
    score_card_deploy.rename(columns={'打分':'变量打分','中间层指标名称':'指标英文'},inplace=True)
    # 选出模型中用到的字段名和其对应的
    model_selected = [i for i in set(list(score_card_deploy['指标英文'])) if str(i)!='nan']
    mlr_selected_vars =  [i for i in set(list(score_card_deploy['输出打分指标名称'])) if str(i)!='nan']

    # 生成数据字典
    var_dict = score_card_deploy.drop_duplicates('指标英文')
    var_dict = var_dict[var_dict['指标英文'].isin(model_selected)]

    # 生成打分时候建模的评分卡
    score_card_modeling = score_card_deploy[score_card_deploy['指标英文'].isin(model_selected)]
    score_card_modeling.ix[len(score_card_modeling)+1,'指标英文'] = 'intercept'
    const_var = [i for i in set(list(score_card_deploy['输出打分指标名称'])) if 'const' in i][0]
    const_value = float(score_card_deploy.ix[score_card_deploy['输出打分指标名称']==const_var,'变量打分'])
    score_card_modeling.ix[score_card_modeling['指标英文']=='intercept','变量打分'] = const_value
    # 生成建模时刻的rebin_spec
    rebin_spec_bin_adjusted_modeling = rebin_spec_bin_adjusted_deploy['numerical']
    rebin_spec_bin_adjusted_modeling.update(rebin_spec_bin_adjusted_deploy['categorical'])

    # 生成变量名称与打分名称对应关系
    modelVar_scoreVar_dict = {}
    for j,i in score_card_deploy[['指标英文','输出打分指标名称']].drop_duplicates(['指标英文']).iterrows():
        if str(i['指标英文'])!='nan':
            modelVar_scoreVar_dict[i['指标英文']] = i['输出打分指标名称']

    # 转换原始数据格式
    try:
        data_origin = data_origin.sort_values(['apply_id','var_code','created_at'],ascending=False)\
        .drop_duplicates(['apply_id','var_code','created_at'])
    except:
        try:
            data_origin = data_origin.sort_values(['apply_id','var_code'],ascending=False)\
            .drop_duplicates(['apply_id','var_code'])
        except:
            pass

    data_origin_pivot = data_origin.pivot(index='apply_id',columns='var_code',values='var_val')

    # 找出变量缺失的用户
    wrong_data = data_origin_pivot[data_origin_pivot.isnull().sum(axis=1)>0]
    wrong_data.reset_index(inplace=True)
    # 找出不缺失打分的用户
    right_data = data_origin_pivot[data_origin_pivot.isnull().sum(axis=1)==0]

    right_data_converted = convert_right_data_type(right_data[model_selected],var_dict)[0]

    # 将数据转换为bin
    bin_obj = mt.BinWoe()
    data_origin_pivot_cat = bin_obj.convert_to_category(right_data_converted[model_selected], var_dict\
                                               , rebin_spec_bin_adjusted_modeling)
    data_origin_pivot_score_all = mt.Performance().calculate_score_by_scrd(data_origin_pivot_cat[model_selected]\
                                                               ,score_card_modeling)
    data_origin_pivot_score_vars = data_origin_pivot_score_all[0]
    data_origin_pivot_score_model = data_origin_pivot_score_all[1]


    data_origin_pivot_cat.columns = [i+'_cat' for i in data_origin_pivot_cat.columns]
    data_origin_pivot_score_vars.columns = [i+'_varScoreBymodel' for i in data_origin_pivot_score_vars.columns]

    data_origin_pivot_all = pd.merge(pd.merge(data_origin_pivot,data_origin_pivot_cat,left_index=True,right_index=True)\
             ,data_origin_pivot_score_vars,left_index=True,right_index=True)

    # 将打分错误的变量对应的申请找出来
    wrong_score_vars = {}
    for i in model_selected:
        wrong_dt =data_origin_pivot_all[data_origin_pivot_all\
                                        .apply(lambda x : float(x[modelVar_scoreVar_dict[i]])!=float(x[i+'_varScoreBymodel']),axis=1)]\
                                        [[i,i+'_cat',i+'_varScoreBymodel',modelVar_scoreVar_dict[i]]]

        wrong_score_vars[i] = wrong_dt.reset_index()


    # 整合线上和线下打分
    mlr_score_vars = [i for i in mlr_selected_vars if 'creditscore' in i][0]
    model_on_off_score = pd.merge(data_origin_pivot_score_model.to_frame('back_score'),data_origin_pivot_all[mlr_score_vars]\
         .to_frame('mlr_score'),left_index=True,right_index=True)

    wrong_model_score = model_on_off_score[model_on_off_score\
                                           .apply(lambda x : float(x['back_score']) != float(x['mlr_score']),axis=1)]
    wrong_model_score.reset_index(inplace=True)

    # 求变量的PSI
    var_dist = []
    for i in model_selected:
        tmp = (data_origin_pivot_cat[i+'_cat'].value_counts())\
        .to_frame('allNum').reset_index()
        tmp['dist'] = tmp['allNum']/len(data_origin_pivot_cat)

        tmp['指标英文'] = i
        tmp.columns = ['分箱','allNum','dist','指标英文']
        var_dist.append(tmp)
    var_dist_data = pd.concat(var_dist)[['指标英文','分箱','allNum','dist']]
    var_dist_data['分箱'] = var_dist_data['分箱'].astype(str)
    var_dist_data_with_score_card = pd.merge(score_card_modeling,var_dist_data,on = ['指标英文','分箱'])
    var_dist_data_with_score_card['PSI'] = var_dist_data_with_score_card.apply(lambda x : (x['dist'] - x['分布占比'])\
                           *log(x['dist']/x['分布占比'])\
                           if x['dist']!=0 and x['分布占比']!=0 else 0\
                           ,axis=1)
    var_dist_data_with_score_card = var_dist_data_with_score_card\
    [['指标分类','指标英文','变量中文','数据类型','分箱','N','分布占比','allNum','dist','PSI']]
    # 求打分的psi
    # 拿到打分的cut_bounds
    point_bounds = mt.BinWoe().obtain_boundaries(model_decile_deploy[u'分箱'])['cut_boundaries']

    mlr_score_dist = (pd.cut(data_origin_pivot_all[mlr_score_vars].astype(float),point_bounds)\
    .value_counts().to_frame('allNum')).reset_index()

    mlr_score_dist['dist'] = mlr_score_dist['allNum']/len(data_origin_pivot_all)
    mlr_score_dist['index'] = mlr_score_dist['index'].astype(str)
    model_decile_deploy['样本分布'] = model_decile_deploy['样本数']/sum(model_decile_deploy['样本数'])
    score_dist_with_decile = pd.merge(model_decile_deploy,mlr_score_dist,left_on='分箱',right_on='index')\
    [['分箱','样本数','样本分布','allNum','dist']]

    score_dist_with_decile['PSI'] = score_dist_with_decile.apply(lambda x : (x['dist'] - x['样本分布'])\
                           *log(x['dist']/x['样本分布'])\
                           if x['dist']!=0 and x['样本分布']!=0 else 0\
                           ,axis=1)

    return wrong_data,wrong_score_vars,wrong_model_score,var_dist_data_with_score_card,score_dist_with_decile

def get_xgboost_tree_leaf_dist_and_badRate(xgboost_model_result,data_xgboost,data_y,y_name):
    '''
    该函数为上线以后统计xgboost上每一颗决策树叶子对应的分布以及逾期率

    Args:

    xgboost_model_result(dict):xgboost输出的模型结果,
    data_xgboost(DataFrame):xgboost衍生完成以后的X变量
    data_y(Series)：y数据
    y_name(str):y变量的名称

    Returns：
    data_all_leaf(DataFrame):所有决策叶子的分布以及逾期率
    '''
    booster = xgboost_model_result['model_final']
    features_name = booster.__dict__['feature_names']
    leafs_tmp_data = booster.predict(xgb.DMatrix(data_xgboost[features_name]),pred_leaf=True)
    leafs_tmp_data = pd.DataFrame(leafs_tmp_data)
    leafs_tmp_data.columns = ['tree_%s'%str(i) for i in leafs_tmp_data.columns]
    leafs_tmp_data.index = data_xgboost.index
    leafs_tmp_data_with_y = pd.merge(leafs_tmp_data,data_y.to_frame(y_name)\
                                     ,left_index=True,right_index=True)
    data_all_leaf = []
    for i in leafs_tmp_data.columns:
        tmp = leafs_tmp_data_with_y.groupby(i)[y_name]\
        .agg([('allNum',len),('badNum',sum),('badRate',lambda x : sum(x)/len(x))])
        tmp['dist'] = tmp['allNum']/tmp['allNum'].sum()
        tmp.reset_index(inplace=True)
        tmp['var_name'] = i
        tmp.rename(columns = {i:'leafNum'},inplace=True)
        data_all_leaf.append(tmp)
    return pd.concat(data_all_leaf)[['var_name','leafNum','allNum','badNum','dist','badRate']].reset_index(drop=True)


def compare_train_OOT_leaf_psi_and_badRate(train_leafs,train_set_name,oot_leafs,oot_set_name):
    '''
    该函数为上线以后统计xgboost在train和oot上每一颗树叶子上的逾期率以及PSI变化

    Args:

    train_leafs(DataFrame):train的叶子信息,
    oot_leafs(DataFrame):oot的叶子信息
    train_set_name(str)：train数据集名称
    oot_set_name(str):oot数据集名称

    Returns：
    all_leafs(DataFrame):所有决策叶子的PSI以及逾期率
    '''
    train_leafs_copy = train_leafs.copy()
    oot_leafs_copy = oot_leafs.copy()
    train_leafs_copy.columns = [i+'_'+train_set_name if i not in ['var_name','leafNum'] else i for i in train_leafs_copy.columns]
    oot_leafs_copy.columns = [i+'_'+oot_set_name if i not in ['var_name','leafNum'] else i for i in oot_leafs_copy.columns]
    all_leafs = pd.merge(train_leafs_copy,oot_leafs_copy,on=['var_name','leafNum'],how='outer')
    all_leafs['psi'] = all_leafs.apply(lambda x : (x['dist'+'_'+train_set_name] - x['dist'+'_'+oot_set_name])\
                           *log(x['dist'+'_'+train_set_name]/x['dist'+'_'+oot_set_name])\
                           if x['dist'+'_'+train_set_name]!=0 and x['dist'+'_'+oot_set_name]!=0 else 0\
                           ,axis=1)
    return all_leafs



def xgb_online_score_validity(data,input_vars,var_dict,var_transform_method,model_booster,model_decile,onLineScoreName):
    '''
    该函数为xgb模型上线以后统计是否存在缺失值未填充、打分错误以及PSI相关函数

    Args:

    data(DataFrame):其中表头必须含有apply_id,var_code,create_at字段,
    大家严格按照部署监控时刻的要求撰写PSI监控时刻的SQL取数则不会出错，如果唯一标识不是apply_id，务必将唯一标识字段名修改为apply_id
    input_vars(list):原始变量
    var_dict(DataFrame)：数据字典
    var_transform_method(json):部署时刻的变量转换文件
    model_booster(booster):模型部署时刻的booster文件
    model_decile(DataFrame):model建模时候的decile
    onLineScoreName:xgb模型线上打分字段名

    Returns：
    data_input_vars_missing(DataFrame):未填充缺失值的所有apply
    data_scored_applys_wrong(DataFrame):打分错误明细
    model_decile_summary(DataFrame):模型分PSI
    '''
    data = data.sort_values(['apply_id','var_code','created_at'],ascending=False)\
    .drop_duplicates(['apply_id','var_code'])

    # 将数据进行列行转换
    data_pivot_all = data.pivot(index='apply_id',columns='var_code',values='var_val')

    # 变量出现空值的apply统计
    data_input_vars_missing = data_pivot_all[data_pivot_all[input_vars].isnull().sum(axis=1)>0]
    data_scored_applys = data_pivot_all[data_pivot_all[input_vars].isnull().sum(axis=1)==0]

    # 进模型前的原始数据
    data_scored_applys_origin = data_scored_applys[input_vars]
    # 将数据转化为字典中的格式
    data_scored_applys_origin = mu.convert_right_data_type(data_scored_applys_origin, var_dict)[0]

    try:
        # 将需要做dummy的变量进行dummy转化
        data_dummys = pd.get_dummies(data_scored_applys_origin[list(var_transform_method['dummy_vars'].keys())]\
                                 ,columns=list(var_transform_method['dummy_vars'].keys()), drop_first=False\
                                 , prefix_sep='DUMMY')
        # 进入模型的数据
        data_in_model = pd.merge(data_scored_applys_origin,data_dummys,left_index=True,right_index=True)
    except:
        data_in_model = data_scored_applys_origin
    data_in_model.rename(columns=var_transform_method['dummy_var_name_map'],inplace=True)

    # 转换成xgb格式
    data_in_model_DMatrix = xgb.DMatrix(data_in_model[model_booster.__dict__['feature_names']])

    data_scored_applys['offLine_prob'] = model_booster.predict(data_in_model_DMatrix)
    data_scored_applys['offLineScore'] = data_scored_applys['offLine_prob']\
    .apply(lambda x : mt.Performance().p_to_score(x))

    # 打分错误的applyid
    data_scored_applys_wrong = data_scored_applys[data_scored_applys[[onLineScoreName,'offLineScore']]\
    .apply(lambda x : float(x[onLineScoreName]) != float(x['offLineScore']),axis=1)]

    data_scored_applys_right = data_scored_applys[data_scored_applys[[onLineScoreName,'offLineScore']]\
    .apply(lambda x : float(x[onLineScoreName]) == float(x['offLineScore']),axis=1)]

    # PSI统计
    point_bounds_XGBRandom = mt.BinWoe().obtain_boundaries(model_decile['分箱'])['cut_boundaries']
    model_decile_curr = pd.cut(data_scored_applys_right[onLineScoreName].astype(float)\
       ,point_bounds_XGBRandom).value_counts().sort_index().to_frame('curr_num').reset_index()

    model_decile_final = pd.concat([model_decile,model_decile_curr],axis=1)
    model_decile_final['建模分布'] = model_decile_final['样本数']/model_decile_final['样本数'].sum()
    model_decile_final['curr_dist'] = model_decile_final['curr_num']/model_decile_final['curr_num'].sum()
    model_decile_final['psi'] = model_decile_final.apply(lambda x : (x['建模分布'] - x['curr_dist'])\
                           *log(x['建模分布']/x['curr_dist'])\
                           if x['curr_dist']!=0 and x['建模分布']!=0 else 0\
                           ,axis=1)
    model_decile_summary = model_decile_final[['分箱','样本数','建模分布','curr_num','curr_dist','psi']]
    return data_input_vars_missing,data_scored_applys_wrong,model_decile_summary



def get_xgb_fscore_with_monthly_PSI(model_result,var_dist_badRate_by_time):
    '''
    该函数为xgb模型变量重要性以及不同时间段的变量PSI

    Args:

    model_result(dict):xgb模型的生成文件
    var_dist_badRate_by_time(DataFrame):变量随时间变化的分布以及逾期率

    Returns：
    xgb模型变量重要性以及不同时间段的变量PSI
    '''
    fscore = pd.Series(model_result['model_final'].get_fscore())\
    .to_frame('feature_importance').sort_values(by='feature_importance',ascending=False)\
    .reset_index().rename(columns={'index':'指标英文'})

    tmp = pd.concat([var_dist_badRate_by_time['basic']['指标英文'],var_dist_badRate_by_time[['sum_PSI']]],axis=1)

    return fscore.merge(tmp,on='指标英文').drop_duplicates().reset_index(drop=True)


def convert_scrd2rebin(score_card):
    '''
    该函数将已经生成的线上部署文档的2_评分卡(logistics和xgb通用)转换为rebin_spec过程，方便进行woe转换
    配合
    bin_obj = mt.BinWoe()；
    X_cat_train = bin_obj.convert_to_category(data[selected], var_dict, rebin_spec)
    woe_iv_df_coarse = bin_obj.calculate_woe_all(X_cat_train, y_train, var_dict, rebin_spec)
    可以将离线数据进行分箱统计，包括逾期率以及分布
    Args:

    score_card(DataFrame):评分卡

    Returns：
    rebin_spec
    '''
    select_numerical = list(score_card[(score_card['中间层指标名称'].notnull())\
                                        &(score_card['数据类型']!='varchar')]['中间层指标名称'].unique())
    select_categorical = list(score_card[(score_card['中间层指标名称'].notnull())\
                                        &(score_card['数据类型']=='varchar')]['中间层指标名称'].unique())
    
    rebin_spec = {}
    
    if len(select_numerical)>=1:
        for i in select_numerical:
            rebin_spec[i] = mt.BinWoe().obtain_boundaries(score_card[score_card['中间层指标名称'] == i].分箱)

    if len(select_categorical)>=1:            
        for i in select_categorical:
            cat_dict = {}
            for j,k in enumerate(score_card[score_card['中间层指标名称']==i].分箱):
                cat_dict[j+1] = k.split(',')
            rebin_spec[i] = cat_dict
        
    return rebin_spec

