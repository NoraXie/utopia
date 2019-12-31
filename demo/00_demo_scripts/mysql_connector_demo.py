'''
0 导包
需要下载 sqlalchemy, pymysql
将anaconda切换到正确的环境后执行下面两行代码
python -m pip install sqlalchemy -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install pymysql -i https://pypi.tuna.tsinghua.edu.cn/simple
'''


import sys
from jinja2 import Template
import pandas as pd
sys.path.append('/Users/zhaoyiming01/Documents/repos/genie')
from utils3.data_io_utils import *
# from sqlalchemy import create_engine
# import configparser
# import traceback
# import pymysql

'''
1 连接mysql数据库工具包
'''
# def connet_db(db_names, conffile='mysql.cfg', cfg_path=-1):
#     """
#     由配置文件读取数据库连接参数, 根据相应配置返回对应数据库的连接.
#     :param db_names(list): list, 配置文件中[]中的名字为database schema.
#     :param conffile(str): mysql.cfg, 默认情况下读取与当前在同一路径下的mysql.cfg文件.
#     :param cfg_path(str): mysql.cfg文件的路径; 默认值为-1,表示不需要传入文件路径.
#     :return (dict):
#         conn_dict
#         根据配置文件中填写的schema数量, 返回对应schema的连接对象, 并逐一按schema名称封装进dict.
#         使用时conn_dict[schema]取出连接对象,以调用sql api.
#         如: conn_dict =
#         {
#             'jr_loan': <sqlalchemy.engine.base.Connection object at 0x114cec4e0>,
#             'pay_loan': <sqlalchemy.engine.base.Connection object at 0x114cdd198>
#         }
#     """
#     conn_dict = {}
#     try:
#         conf = configparser.ConfigParser()
#         if cfg_path==-1:
#             conffile = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'mysql.cfg')
#             print(conffile)
#         else:
#             conffile = os.path.join(cfg_path, 'mysql.cfg')
#         with open(conffile, 'r') as cfg:
#             conf.read_file(cfg)
#             cfg.closed
#         for idx, db_name in enumerate(db_names):
#             username = conf.get(db_name, 'username')
#             password = conf.get(db_name, 'password')
#             url = conf.get(db_name, 'url')
#             port = conf.get(db_name, 'port')
#             database = conf.get(db_name, 'database')
#             connect_str = Template("mysql+pymysql://{{username}}:{{password}}@{{url}}:{{port}}/{{database}}").render(
#                 username=username, password=password, url=url, port=port, database=database)
#             # print(connect_str)
#             engine = create_engine(connect_str, echo=True)
#             connector = engine.connect()
#             conn_dict[db_name] = connector
#         return conn_dict
#     except Exception as e:
#         # 如果运行出错, 这行代码用于打印异常的堆栈信息, 通常是红色的文字
#         traceback.print_exc()
#         return 400


'''
2 demo
'''
#2.1 从mysql.cfg文件看来, 有三个schema, 分别是jr_loan, pay_loan
db_names = ['jr_loan','pay_loan']

#2.2 连接数据库
connectors = connet_db(db_names)

# 单独使用某一个库的连接, 可先通过db_name取出连接
jr_loan_conn = connectors['jr_loan']
pay_loan_conn = connectors['pay_loan']

#2.3 使用stock_conn这个连接查询数据库, 记得将table换成自己的
get_all_tickers_sql = """
select * from loan_credit_apply where channel_code='{{symbol}}' limit 1
"""

all_tickers = pd.read_sql(Template(get_all_tickers_sql).render(symbol='MI'), jr_loan_conn)
print(all_tickers.head())


#2.4 最后如果不再使用, 或者10分钟以上不用该连接, 记得关闭数据库连接
for db_name, conn in connectors.items():
    conn.closed