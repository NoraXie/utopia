"""
Python 3.6

1、部署logistics评分卡
2、部署xgboost相关文件

"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from jinja2 import Template
import utils3.misc_utils as mu
from utils3.data_io_utils import *
import utils3.metrics as mt


# 测试通过

def generate_logistics_scorecard_deployment_documents(model_label,
                                            RESULT_PATH,score_card,model_decile,
                                            eda_table,
                                            coarse_classing_rebin_spec,
                                            production_name_map={}):

    """
    生成逻辑回归评分卡部署Excel文件，线上部署json文件，线上部署测试用例

    Args
    model_label (str): 模型名称
    RESULT_PATH (str): 结果路径
    score_card (dataFrame): 建模评分卡
    model_decile(dataFrame): 建模decile
    eda_table (dataFrame): EDA结果
    coarse_classing_rebin_spec (dict): 建模分箱边界文件
    production_name_map (dict): 当有指标英文线上的名字和线下建模时不一样时传入。key=建模时英文名
        value=线上英文名。 default={} 默认建模时和线上没有命名不一致的情况
    """
    # 创建部署路径
    if not os.path.exists(os.path.join(RESULT_PATH, 'deployment')):
        os.makedirs(os.path.join(RESULT_PATH, 'deployment'))

    DEPLOY_PATH = os.path.join(RESULT_PATH, 'deployment')

    score_card_production = score_card.rename(columns={'指标英文': '中间层指标名称', '变量打分': '打分'})
    score_card_production['中间层指标名称'] = score_card_production['中间层指标名称'].replace(production_name_map)
    score_card_production.insert(2, '输出打分指标名称',
                score_card_production['中间层指标名称'].apply(lambda x: 'mlr_' + x + '_scrd_' + model_label))

    score_card_production.loc[:, '输出打分指标名称'] = score_card_production.loc[:, '输出打分指标名称']\
                                                        .replace('mlr_intercept_scrd_'+model_label, 'mlr_const_scrd_'+model_label)

    score_card_production.loc[score_card_production['中间层指标名称']=='intercept', '指标中文'] = '截距分'
    score_card_production.loc[score_card_production['中间层指标名称']=='intercept', '中间层指标名称'] = None

    score_card_production = score_card_production.append({'输出打分指标名称': 'mlr_creditscore_scrd_'+model_label}, ignore_index=True)
    score_card_production = score_card_production.append({'输出打分指标名称': 'mlr_prob_scrd_'+model_label}, ignore_index=True)
    score_card_production.insert(score_card_production.shape[1], '是否手动调整', None)
    score_card_production.insert(score_card_production.shape[1], 'backscore分布占比', None)
    score_card_production.insert(score_card_production.shape[1], '基准psi', None)
    score_card_production.insert(score_card_production.shape[1], 'psi预警delta', None)

    selected_variables = [i for i in score_card['指标英文'].unique() if i != 'intercept']
    model_eda = eda_table.loc[eda_table['指标英文'].isin(selected_variables)]

    writer = pd.ExcelWriter(os.path.join(DEPLOY_PATH, '%s部署评分卡.xlsx' % model_label))
    score_card_production.to_excel(writer, '2_模型评分卡', index=False)
    model_decile.to_excel(writer, '3_模型decile', index=False)
    model_eda.to_excel(writer, '4_模型EDA', index=False)
    writer.save()
    logging.info("""第六步部署：生成逻辑回归评分卡部署文档。
    1. 模型部署Excel文档存储于：%s
    2. 需添加『0_文档修订记录』、『1_信息总览』页面。详见其他正式部署文档文件。并存储于『/Seafile/模型共享/模型部署文档/』相应文件夹中
    """ % os.path.join(DEPLOY_PATH, '%s部署评分卡.xlsx' % model_label))

    bin_to_score_json = mu.process_bin_to_score(score_card)
    for original, new_name in production_name_map.items():
        coarse_classing_rebin_spec[new_name] = coarse_classing_rebin_spec.pop(original)
        bin_to_score_json[new_name] = bin_to_score_json.pop(original)

    rebin_spec_json = mu.process_rebin_spec(coarse_classing_rebin_spec, score_card, selected_variables)

    save_data_to_json(rebin_spec_json, DEPLOY_PATH, '%s_selected_rebin_spec.json' % model_label)
    save_data_to_json(bin_to_score_json, DEPLOY_PATH, '%s_bin_to_score.json' % model_label)
    logging.info("""第六步部署：生成逻辑回归评分卡部署文档。
    线上部署配置文件存储于%s路径下
    1. %s
    2. %s
    """ % (DEPLOY_PATH,
           '%s_selected_rebin_spec.json' % model_label,
           '%s_bin_to_score.json' % model_label
          ))

    writer_old = pd.ExcelWriter(os.path.join(DEPLOY_PATH, '%s_testcases_old.xlsx' % model_label))
    writer_new = pd.ExcelWriter(os.path.join(DEPLOY_PATH, '%s_testcases.xlsx' % model_label))
    local_test_case = mu.produce_http_test_case(score_card_production, model_label, model_label,local=True)
    qa_test_case = mu.produce_http_test_case(score_card_production, model_label, model_label,local=False)
    auto_test_case = mu.auto_deplyment_testcases(score_card_production, model_label)
    local_test_case.to_excel(writer_old,'local',index=False)
    qa_test_case.to_excel(writer_old,'qa',index=False)
    auto_test_case.to_excel(writer_new, index=False)

    writer_old.save()
    writer_new.save()
    logging.info("""第六步部署：生成逻辑回归评分卡部署文档。
    线上部署测试用例存储于 %s
    """ % os.path.join(DEPLOY_PATH, '%s_testcases.xlsx' % model_label))


def generate_xgb_testcases(model_label, x_with_apply_id, var_dict, RESULT_PATH, model_result=None, is_backscoring=False):
    '''
    :param model_label(str): 模型命名, 发布模型时定义的模型名称
    :param x_with_apply_id(DataFrame): X变量, 必须有apply_id, apply_id务必设置为index
    :param var_dict(DataFrame): 变量字典
    :param model_result(dict): 建模时的模型结果, 建模时的[模型结果.pkl]. 必须有p_train. 务必保证model_result['p_train']的索引为apply_id,model_result['p_test']同理
    :param is_backscoring(boolean): True 用于回溯打分, 不需要建模样本的打分结果; False 用于建模样本生成建模样本的测试用例, 需同时传入model_result
    :return:
    '''
    def auto_xgb_testcase(test_data, modelName):
        '''
        :param test_data(dict):
        :param modelName(str):
        :return :json dict
        '''
        strs = ''' {"modelName":"%s","productName":"%s","applyId": 123, ''' % (modelName, modelName)
        for i, j in test_data.items():
            strs = strs + '"%s"' % i + ':"%s"' % j + ', '
        final = strs[:-2] + '''} '''
        return final

    x_columns = list(set((var_dict['指标英文']).unique()).intersection(set(x_with_apply_id.columns.values)))
    x_columns
    X = x_with_apply_id[x_columns]
    X = mu.process_missing(X, var_dict, verbose=True)
    X = mu.convert_right_data_type(X, var_dict)[0]

    if is_backscoring:
        X['origin_features'] = X.to_dict('records')
        X['var_param'] = X.apply(lambda row: auto_xgb_testcase(row['origin_features'],model_label),axis=1)
        test_cases = X[['var_param']]
        test_cases['offline_model_score'] = -8887
    else:
        model_sample_score = pd.concat(
            [model_result['p_train'].apply(mt.Performance().p_to_score).to_frame('offline_model_score'),
             (model_result['p_test'].apply(mt.Performance().p_to_score).to_frame('offline_model_score'))])
        model_sample_x_with_score = model_sample_score.merge(x_with_apply_id, left_index=True, right_index=True,how='inner')
        model_sample_x_with_score['origin_features'] = model_sample_x_with_score[model_sample_x_with_score.columns.difference(['offline_model_score'])].to_dict('records')

        model_sample_x_with_score['var_param'] = model_sample_x_with_score.apply(lambda row: auto_xgb_testcase(row['origin_features'], model_label), axis=1)
        test_cases = model_sample_x_with_score[['offline_model_score', 'var_param']]

    result_df = test_cases[['offline_model_score','var_param']]
    result_df.reset_index(inplace=True)
    result_df.columns = ['applyId','offline_model_score','var_param']
    writer = pd.ExcelWriter(os.path.join(RESULT_PATH, '%s_testcases.xlsx' % model_label))
    result_df.to_excel(writer, index=False)
    writer.save()
    return

def generate_xgb_deployment_documents(model_label,eda_table,model_spec,xgb_importance_score,
                                      var_dict,model_decile,model_result,RESULT_PATH,test_case_data,test_nums
                                      ,woe_iv_df_coarse, production_name_map={}):
    """
    生成XGBoost模型部署文档

    Args
    model_label (str): 模型名称
    eda_table (dataFrame): EDA结果
    model_spec (dataFrame): XGBoost输出数据和分箱明细.pkl 文件
    xgb_importance_score (dataFrame): xgboost模型变量重要性排序
    var_dict (dataFrame): 数据字典
    model_decile(dataFrame): 建模decile
    model_result (dict): 模型结果
    RESULT_PATH (str): 结果路径
    test_case_data(dataFrame): 生成testcase的测试数据, 如果用于部署, 请确保apply_id为index
    test_nums(int):生成测试用例的数量
    woe_iv_df_coarse(dataFrame): 建模粗分箱结果
    production_name_map (dict): 当有指标英文线上的名字和线下建模时不一样时传入。key=建模时英文名
        value=线上英文名。 default={} 默认建模时和线上没有命名不一致的情况
    """
    if not os.path.exists(os.path.join(RESULT_PATH, 'deployment')):
        os.makedirs(os.path.join(RESULT_PATH, 'deployment'))

    DEPLOY_PATH = os.path.join(RESULT_PATH, 'deployment')

    eda_table = eda_table
    model_spec = model_spec
    rebin_spec = model_spec['rebin_spec']
    bin_to_label = model_spec['bin_to_label']
    dummy_var_name_map = model_spec['dummy_var_name_map']

    impt_writer = pd.ExcelWriter(os.path.join(DEPLOY_PATH, '%s_variable_importance.xlsx' % model_label))
    xgb_importance_score.to_excel(impt_writer,'local',index=False)
    impt_writer.save()

    xgb_importance_score = xgb_importance_score
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
        # 这一步很重要，要将字典里边的变量名也改掉，不然rebin_spec_json = mu.process_rebin_spec(rebin_spec, var_dict, num_variables+bin_variables)会出错
        if original in list(var_dict['指标英文']):
            var_dict['指标英文'].replace({original:new_name},inplace=True)

    xgb_importance_score['中间层指标名称'] = xgb_importance_score['中间层指标名称'].replace(production_name_map)

    xgb_importance_score = var_dict[['数据源', '指标英文', '指标中文', '数据类型']]\
                                .rename(columns={'指标英文':'中间层指标名称'})\
                                .merge(xgb_importance_score, on='中间层指标名称', how='right')

    xgb_importance_score.insert(5, '输出打分指标名称',
                xgb_importance_score['XGB衍生入模名称'].apply(lambda x: 'mlr_' + str(x) + '_xgb_' + model_label))


    xgb_importance_score = xgb_importance_score.append({'输出打分指标名称': 'mlr_creditscore_xgb_'+model_label}, ignore_index=True)
    xgb_importance_score = xgb_importance_score.append({'输出打分指标名称': 'mlr_prob_xgb_'+model_label}, ignore_index=True)

    model_decile = model_decile

    selected_variables = xgb_importance_score['建模时指标名称'].unique()
    model_eda = eda_table.loc[eda_table['指标英文'].isin(selected_variables)].copy()
    model_eda['指标英文'] = model_eda['指标英文'].replace(production_name_map)

    woe_iv_df_coarse_copy = woe_iv_df_coarse.copy()
    if len(production_name_map)>0:
        for j,k in production_name_map.items():
            woe_iv_df_coarse_copy = woe_iv_df_coarse_copy.replace(j,k)

    xgb_importance_score.rename(columns={'数据源':'指标分类'}, inplace=True)
    xgb_importance_score_bin_result = pd.merge(xgb_importance_score.drop_duplicates('中间层指标名称')[['指标分类','中间层指标名称','XGB衍生入模名称','输出打分指标名称','指标用于Split数据的数量'\
     ,'指标用于Split数据的数量占比','XGB变量转换类型','建模时指标名称','建模时XGB衍生入模名称']]\
         ,woe_iv_df_coarse_copy,left_on='中间层指标名称',right_on='指标英文',how='left')\
    [['指标分类','中间层指标名称','指标中文','XGB衍生入模名称','输出打分指标名称','指标用于Split数据的数量'\
     ,'指标用于Split数据的数量占比','XGB变量转换类型','建模时指标名称','建模时XGB衍生入模名称'\
    ,'数据类型','指标类型','分箱','分箱对应原始分类','N','分布占比','WOE','逾期率']]

    xgb_importance_score_bin_result['分箱'] = xgb_importance_score_bin_result.apply(lambda x : x['分箱对应原始分类'] if x['分箱对应原始分类'] else x['分箱'],axis=1)
    xgb_importance_score_bin_result.drop(['分箱对应原始分类'],axis=1,inplace=True)

    writer = pd.ExcelWriter(os.path.join(DEPLOY_PATH, '%s部署文档.xlsx' % model_label))
    xgb_importance_score_bin_result.to_excel(writer, '2_模型变量重要性排序及分箱统计', index=False)
    model_decile.to_excel(writer, '3_模型decile', index=False)
    model_eda.to_excel(writer, '4_模型EDA', index=False)
    writer.save()
    logging.info("""第六步部署：生成XGBoost部署文档。
    1. 模型部署Excel文档存储于：%s
    2. 需添加『0_文档修订记录』、『1_信息总览』页面。详见其他正式部署文档文件。并存储于『/Seafile/模型共享/模型部署文档/』相应文件夹中
    """ % os.path.join(DEPLOY_PATH, '%s部署文档.xlsx' % model_label))


    model_result = model_result
    derive_name_map = dict(zip(xgb_importance_score['建模时XGB衍生入模名称'], xgb_importance_score['XGB衍生入模名称']))
    xgbmodel = model_result['model_final']

    var_list = []
    for i in xgbmodel.__dict__['feature_names']:
        try:
            var_list.append(derive_name_map[i])
        except:
            var_list.append(i)
    xgbmodel.__dict__['feature_names'] = var_list

    if len(production_name_map)>0:
        var_list_copy = []
        for i in var_list:
            for j,k in production_name_map.items():
                r = i.replace(j,k)
            var_list_copy.append(r)
        xgbmodel.__dict__['feature_names'] = var_list_copy

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

    production_dummy_var_name_map = {}
    for new_dummy_name, old_dummy_name in dummy_var_name_map.items():
        if new_dummy_name != old_dummy_name:
            if new_dummy_name.split('DUMMY')[0] in production_name_map:
                var_name = new_dummy_name.split('DUMMY')[0]
                prod_var_name = production_name_map[var_name]
                production_dummy_var_name_map[old_dummy_name.replace(var_name, prod_var_name)] = new_dummy_name.replace(var_name, prod_var_name)
            else:
                production_dummy_var_name_map[old_dummy_name] = new_dummy_name

    var_transform_method['dummy_var_name_map'] = production_dummy_var_name_map

    save_data_to_json(rebin_spec_json, DEPLOY_PATH, '%s_selected_rebin_spec.json' % model_label)
    save_data_to_json(bin_to_label, DEPLOY_PATH, '%s_bin_to_label.json' % model_label)
    save_data_to_python2_pickle(xgbmodel, DEPLOY_PATH, '%s_xgbmodel.pkl' % model_label)
    save_data_to_json(var_transform_method, DEPLOY_PATH, '%s_var_transform_method.json' % model_label)

    ''' 生成模型的testcase '''
    if test_nums<len(test_case_data):
        test_case_data_final = test_case_data[:test_nums]
    else:
        test_case_data_final = test_case_data.copy()
    if len(production_name_map)>0:
        columns = []
        for i in test_case_data_final.columns:
            for j,k in production_name_map.items():
                r = i.replace(j,k)
            columns.append(r)
        test_case_data_final.columns = columns

    offline_model_score = pd.concat([model_result['p_train'].apply(mt.Performance()\
                                  .p_to_score).to_frame('offline_model_score'),\
                                    model_result['p_test'].apply(mt.Performance()\
                                  .p_to_score).to_frame('offline_model_score')])

    model_used_origin_features = [i for i in list(set(xgb_importance_score['中间层指标名称'])) if str(i)!='nan' and i !=None]
    test_case_data_final_with_xgbScore = pd.merge(test_case_data_final[model_used_origin_features], offline_model_score,left_index=True,right_index=True)

    def get_xgb_testcase(test_data,modelName,productName):
        strs = ''' curl -d 'varparams={"modelName":"%s","productName":"%s","applyId": 123, '''%(modelName,productName)
        for i,j in test_data.items():
            strs =strs + '"%s"'%i+':"%s"'%j+', '
        final = strs[:-2] + '''}' -X POST http://localhost:8080/modelInvoke'''
        return final
    test_case_data_final_with_xgbScore['test_case'] = test_case_data_final_with_xgbScore[test_case_data_final_with_xgbScore.columns.difference(['offline_model_score'])].to_dict('records')
    test_case_data_final_with_xgbScore['curl_script'] = test_case_data_final_with_xgbScore['test_case'].apply(get_xgb_testcase,args=(model_label,model_label ))
    test_case_data_final_with_xgbScore[['offline_model_score','curl_script']].to_csv(os.path.join(DEPLOY_PATH,'%s_xgbmodel_testcase.csv'%model_label),index=None)

    '''为秒算模型部署平台生成测试用例'''
    generate_xgb_testcases(model_label, test_case_data_final, var_dict, DEPLOY_PATH, model_result)

    logging.info("""第六步部署：生成XGBoost部署文档。
    线上部署配置文件存储于%s路径下
    1. %save_data_dict_to_pickle
    2. %s
    3. %s
    4. %s
    """ % (DEPLOY_PATH,
           '%s_selected_rebin_spec.json' % model_label,
           '%sbin_to_label.json' % model_label,
           '%s_var_transform_method.json' % model_label,
           '%s_xgbmodel.pkl' % model_label,
          ))