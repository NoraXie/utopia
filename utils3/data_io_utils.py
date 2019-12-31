# encoding=utf8
import os
import pickle as pickle
import simplejson as json
from jinja2 import Template
import codecs
import smtplib # python3 自带
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders
import traceback
import pymysql
from sqlalchemy import create_engine
# from prestodb.client import *
# import MySQLdb
import configparser

conf = configparser.ConfigParser()
# DATABASE_CONFIGURE_FILE = os.path.join(os.path.split(os.path.realpath('mysql.cfg'))[0],'mysql.cfg')
DATABASE_CONFIGURE_FILE = os.path.join(os.path.split(os.path.realpath(__file__))[0],'mysql.cfg')
with open(DATABASE_CONFIGURE_FILE, 'r') as cfgfile:
    conf.readfp(cfgfile)

'''
1 连接mysql数据库工具包
'''
def connet_db(db_names, conffile='mysql.cfg', cfg_path=-1):
    """
    由配置文件读取数据库连接参数, 根据相应配置返回对应数据库的连接.
    :param db_names(list): list, 配置文件中[]中的名字为database schema.
    :param conffile(str): mysql.cfg, 默认情况下读取与当前在同一路径下的mysql.cfg文件.
    :param cfg_path(str): mysql.cfg文件的路径; 默认值为-1,表示不需要传入文件路径.
    :return (dict):
        conn_dict
        根据配置文件中填写的schema数量, 返回对应schema的连接对象, 并逐一按schema名称封装进dict.
        使用时conn_dict[schema]取出连接对象,以调用sql api.
        如: conn_dict =
        {
            'stock': <sqlalchemy.engine.base.Connection object at 0x114cec4e0>,
            'mysql': <sqlalchemy.engine.base.Connection object at 0x114cdd198>,
            'sys': <sqlalchemy.engine.base.Connection object at 0x114d00278>
        }
    """
    conn_dict = {}
    try:
        # conf = configparser.ConfigParser()
        if cfg_path==-1:
            conffile = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'mysql.cfg')
        else:
            conffile = os.path.join(cfg_path, 'mysql.cfg')
        with open(conffile, 'r') as cfg:
            conf.read_file(cfg)
            cfg.closed
        for idx, db_name in enumerate(db_names):
            username = conf.get(db_name, 'username')
            password = conf.get(db_name, 'password')
            url = conf.get(db_name, 'url')
            port = conf.get(db_name, 'port')
            database = conf.get(db_name, 'database')
            connect_str = Template("mysql+pymysql://{{username}}:{{password}}@{{url}}:{{port}}/{{database}}").render(
                username=username, password=password, url=url, port=port, database=database)
            # print(connect_str)
            engine = create_engine(connect_str)
            connector = engine.connect()
            conn_dict[db_name] = connector
        return conn_dict
    except Exception as e:
        # 如果运行出错, 这行代码用于打印异常的堆栈信息, 通常是红色的文字
        traceback.print_exc()
        return 400


def presto_read_sql_df(sql):
    request = PrestoRequest(host=conf.get('presto', 'PRESTO_HOST'), port=conf.get('presto', 'PRESTO_PORT')\
                    , user=conf.get('presto', 'PRESTO_USER'), source=conf.get('presto', 'PRESTO_PASSWORD'))
    query = PrestoQuery(request, sql)
    rows = list(query.execute())
    columns = [col['name'] for col in query._columns]
    df = pd.DataFrame(rows, columns=columns)
    return df


def presto_upload_data(sql):
    """
    If sql is table creation, then rows = [[True]] is succeed.
    If sql is insert query, then rows = [[4]] where 4 is changeable according to
    number of rows inserted
    """
    request = PrestoRequest(host=conf.get('presto', 'PRESTO_HOST'), port=conf.get('presto', 'PRESTO_PORT')\
                    , user=conf.get('presto', 'PRESTO_USER'), source=conf.get('presto', 'PRESTO_PASSWORD'))
    query = PrestoQuery(request, sql)
    try:
        rows = list(query.execute())
        return 'success'
    except:
        print("upload failed, check query")
        return 'failure'


def save_data_to_pickle(obj, file_path, file_name):
    file_path_name = os.path.join(file_path, file_name)
    with open(file_path_name, 'wb') as outfile:
        pickle.dump(obj, outfile)


def save_data_to_python2_pickle(obj, file_path, file_name):
    """
    python2默认protocol=0，最大可取2，python2存储过程中protocol=2，则python3可读
    python3默认protocol=3，最大可取4，python3存储过程中protocol=0，则python2可读
    """
    file_path_name = os.path.join(file_path, file_name)
    with open(file_path_name, 'wb') as outfile:
        pickle.dump(obj, outfile,protocol=0)


def load_data_from_pickle(file_path, file_name):
    file_path_name = os.path.join(file_path, file_name)
    with open(file_path_name, 'rb') as infile:
        result = pickle.load(infile)
    return result

def load_data_from_python2_pickle(file_path, file_name):
    """
    python2与python3在编码格式上不同，python3load过程默认的encoding='ASCII'
    """
    file_path_name = os.path.join(file_path, file_name)
    with open(file_path_name, 'rb') as infile:
        result = pickle.load(infile,encoding='iso-8859-1')
    return result


def save_data_dict_to_pickle(data_dict, file_path, file_name):
    for vintage, sub_data in list(data_dict.items()):
        print(vintage)
        file_path_name = os.path.join(file_path, vintage + '_' + file_name)
        with open(file_path_name, 'wb') as outfile:
            pickle.dump(sub_data, outfile)


def load_data_dict_to_pickle(data_dict_keys, file_path, file_name):
    # data_dict_keys = ['201612', '201701', '201610', '201611', '201607', '201606', '201609', '201608', 'additional']
    result = {}
    for vintage in data_dict_keys:
        file_path_name = os.path.join(file_path, vintage + '_' + file_name)
        with open(file_path_name, 'rb') as infile:
            result[vintage] = pickle.load(infile)
    return result


def save_data_to_json(data_dict, file_path, file_name):
    with codecs.open(os.path.join(file_path, file_name), 'wb', encoding='utf-8') as outfile:
         json.dump(data_dict, outfile, ensure_ascii=False, indent='\t')


def load_data_from_json(file_path, file_name):
    with codecs.open(os.path.join(file_path, file_name), 'rb', encoding='utf-8') as infile:
        data_dict = json.load(infile)
    return data_dict


def send_email(subject, email_content, recipients=['xiangyu.hu@pintec.com'], attachment_path='', attachment_name=[]):
    sender = conf.get('email','EMAIL_USER')
    pwd = conf.get('email', 'EMAIL_PASSWORD')
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ", ".join(recipients)
    msg['Subject'] = subject
    main_content = MIMEText(email_content, 'plain', 'utf-8')
    msg.attach(main_content)
    ctype = 'application/octet-stream'
    maintype, subtype = ctype.split('/', 1)
    # 附件-文件
    if len(attachment_name) > 0:
        for attachment_name1 in attachment_name:
            attachment_path_name = os.path.join(attachment_path, attachment_name1)
            file = MIMEBase(maintype, subtype)
            file.set_payload(open(attachment_path_name, 'rb').read())
            file.add_header('Content-Disposition', 'attachment', filename=attachment_name1)
            encoders.encode_base64(file)
            msg.attach(file)

    # 发送
    smtp = smtplib.SMTP()
    smtp.connect(conf.get('email', 'EMAIL_HOST'), 25)
    smtp.login(sender, pwd)
    smtp.sendmail(sender, recipients, msg.as_string())
    smtp.quit()
    print('success')
