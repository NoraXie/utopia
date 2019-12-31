#!/usr/bin/python
#coding:utf-8
import os
import collections
import pandas as pd
import json
model_dict = collections.OrderedDict()
basedir = os.path.abspath(os.path.dirname(__file__))

''' 配置查询PSI以及KS数据相关的sql '''
Sql_templates = {'PSI':'''select * from hive_pro.hdp_data_sd.hdp_s_model_psi_mid_monitoring where model_name = '{{model_name}}' ''',
                'KS':'''select * from hive_pro.hdp_data_sd.hdp_s_model_ks_mid_monitoring where model_name = '{{model_name}}' ''',
                'PSI_with_product': '''select * from hive_pro.hdp_data_sd.hdp_s_model_psi_mid_monitoring where model_name = '{{model_name}}' and product_name='{{product_name}}' ''',
                'KS_with_product': '''select * from hive_pro.hdp_data_sd.hdp_s_model_ks_mid_monitoring where model_name = '{{model_name}}' and product_name='{{product_name}}' '''}

''' 配置模型列表 '''
model_dict[u'个贷自营'] = ['XJD_A_V4.2_UTNZ@walletH5Credit','XJD_A_V4.2_UTNZ@walletAppCredit','XJD_A_V5_FID7@walletH5Credit'\
,'XJD_A_V5_FID7@walletAppCredit','XJD_A_V5_FID15N700@walletH5Credit','XJD_A_V5_FID15N700@walletAppCredit'\
,'XJD_A_V5_FID15N500@walletH5Credit','XJD_A_V5_FID15N500@walletAppCredit', 'XJD_B_V1','XJD_B_V2'\
,'XJD_A_V5_MOB3@walletAppCredit','XJD_A_V5_MOB3@walletH5Credit','XJD_A_V5_JUPITER@walletAppCredit','XJD_A_V5_JUPITER@walletH5Credit']
model_dict[u'个贷渠道'] = ['JQH_B_V1_VAR9','JQH_B_V1_VAR7','JQH_B_V1_VAR6','JQH_B_V1_VAR4','JQH_A_V2.2_R1','JQH_A_R2_V5_FID7']
model_dict[u'大白'] = ['DB_A_CR_V4_6vars','DB_A_V5_nolist','DB_A_V5_S2_var6']
model_dict[u'分期'] = ['YZF_A_V5']
model_dict[u'小企业'] = ['LSD_SQB_A_V2']
# model_dict[u'其他'] = ['JM_A_V2']

''' 获取模型配置文件 '''
models_config_df = pd.read_csv('./models_config.csv',encoding='GBK')
models_config_df.index = models_config_df.model_name
models_config_df.fillna('',inplace=True)
models_config = json.loads(models_config_df.to_json(orient='index'))

''' 获取用户配置文件 '''
user_config_df = pd.read_csv('./user_config.csv',encoding='GBK')
user_config_df.index = user_config_df.username
user_config = json.loads(user_config_df.to_json(orient='index'))

''' 程序其他配置文件 '''
class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')
