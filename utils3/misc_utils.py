# encoding=utf8
import os
import json
import hashlib
import copy
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn import preprocessing

def generate_md5(str):
    # 创建md5对象
    hl = hashlib.md5()
    # Tips
    # 此处必须声明encode
    # 否则报错为：hl.update(str)    Unicode-objects must be encoded before hashing
    hl.update(str.encode(encoding='utf-8'))
    return hl.hexdigest()


def convert_unicode_to_time(x_unicode):
    if x_unicode:
        return datetime.strptime(str(x_unicode),'%Y-%m-%d')
    else:
        return None

def date_diff(day1, day2):
    day1 = day1.apply(convert_unicode_to_time)
    day2 = day2.apply(convert_unicode_to_time)
    difference = (day2 - day1).astype('timedelta64[D]')
    return difference


def convert_numeric(val):
    try:
        val = float(val)
    except:
        pass
    return val

def cal_sample_weight(y_train, class_weights='balanced'):
    if class_weights=='balanced':
        from sklearn.utils import class_weight
        class_weights = list(class_weight.compute_class_weight('balanced',np.unique(y_train),y_train))

        sample_weight = np.ones(y_train.shape[0], dtype = 'float')
        for i, val in enumerate(y_train):
            sample_weight[i] = class_weights[val-1]

        return sample_weight
    else:
        return None

def find_continuous(val):
    if isinstance(val, str):
        return 'str'
    elif isinstance(val, float):
        return 'float'
    elif isinstance(val, str):
        return 'unicode'
    elif isinstance(val, int):
        return 'int'


# convert to the right data type
def convert_right_data_type(data, var_dict, quick_mode=False):
    """
    Convert to the right data type, have to apply it after processing the missing value.
    Even though the data are converted to the correct type before saving, reading
    it back will change it. For example, some categorical variables are coded
    as integers, they should be string type for usage, but reading it back, it
    will be loaded in as integers.

    Args:
    data (pd.DataFrame): X数据, 已经处理过缺失值
    var_dict(pd.DataFrame): 变量字典
    quick_mode (bool): default=False. If set True, 将会批量转化数据类型，如果某一列
        有问题，则数据整体将会转化失败。

    Returns:
    failed_conversion (list): if quick_mode=False, 转换失败的变量名list

    """
    data = data.copy()

    cols = data.columns.values
    failed_conversion = []

    if quick_mode:
        col_types = var_dict.loc[var_dict['指标英文'].isin(cols), ['指标英文', '数据类型']]
        for col_type in ['varchar', 'float', 'integer']:
            onetype_cols = col_types.loc[col_types['数据类型']==col_type, '指标英文'].tolist()
            try:
                if col_type == 'varchar':
                    data[onetype_cols] = data[onetype_cols].astype(str)
                elif col_type == 'float':
                    data[onetype_cols] = data[onetype_cols].astype(float)
                elif col_type == 'integer':
                    data[onetype_cols] = data[onetype_cols].astype(int)
            except:
                print("one column may have problem, so conversion for type %s failed" % col_type)

        return data

    else:
        cols = list(set(cols).intersection(set(var_dict['指标英文'].tolist())))
        for col in cols:
            right_type = var_dict.loc[var_dict['指标英文']==col, '数据类型'].iloc[0]
            non_missing_data = data.loc[~data[col].isin([-8888, -9999, -8887]), col]
            if non_missing_data.empty:
                current_type = right_type
            else:
                current_type = str(type(data.loc[~data[col].isin([-8888, -9999, -8887]), col].iloc[0]))

            if right_type == 'varchar':
                if 'float' in current_type:
                    try:
                        data[col] = data[col].astype(int).astype(str)
                    except:
                        print(col + ' conversion failed')
                        failed_conversion.append(col)
                else:
                    try:
                        data[col] = data[col].astype(str)
                    except:
                        print(col + ' conversion failed')
                        failed_conversion.append(col)
                if 'object' in str(data[col].dtype):
                    pass
                else:
                    print(col + ' conversion failed. Type should be object, but is still %s' % str(data[col].dtype))
                    failed_conversion.append(col)

            if right_type == 'float':
                try:
                    data[col] = data[col].astype(float)
                except:
                    print(col + ' conversion failed')
                    failed_conversion.append(col)

                if 'float' in str(data[col].dtype):
                    pass
                else:
                    print(col + ' conversion failed. Type should be float, but is still %s' % str(data[col].dtype))
                    failed_conversion.append(col)

            if right_type == 'integer':
                try:
                    data[col] = data[col].astype(int)
                except:
                    print(col + ' conversion failed')
                    failed_conversion.append(col)

                if 'int' in str(data[col].dtype):
                    pass
                else:
                    print(col + ' conversion failed. Type should be int, but is still %s' % str(data[col].dtype))
                    failed_conversion.append(col)

        return data, np.unique(failed_conversion)
    
def label_encode(x_train):
    """
    将原始分类变量用数字编码

    Args:
    x_train (pd.Series): 原始数值的分类变量

    Returns:
        (tuple)
    x_train_encoded (pd.Series): 数字编码后的变量
    """
    le = preprocessing.LabelEncoder()
    x_train_encoded = le.fit_transform(x_train)
    return pd.Series(x_train_encoded, index=x_train.index)

def process_missing(X, var_dict, known_missing={}, downflagmap={}, verbose=True):
    """
    处理缺失值。如果某个数据源整条数据都没有就算是完整缺失未查得，这些被标成-9999{JiangHong：这里有问题？应该是-8888}。
    有些是可查得，但是个别变量值缺失。这些缺失被label为-8887。

    Args:
    X (pd.DataFrame): 原始数值的分类变量
    var_dict (pd.DataFrame): 标准数据字典，需包含数据源，指标英文两列。
    known_missing (dict): 已知的代表缺失的值以及想要替换成的值。格式为：
        {-1: -9999, -9999999: -8887}
    downflagmap (dict): 中间层有些数据源有downflag字段用来标注是否宕机，是否查无此人等。
        格式为：
        {'Anrong_DownFlag': {1: -9999}, 'Tongdun_DownFlag': {1: -9999}}

    Returns:
    new_X (pd.DataFrame): 填补后的x
    """

    if '数据源' not in var_dict.columns and '指标英文' not in var_dict.columns:
        raise 'Check var_dict column names'

    X = X.replace('nan', np.nan)\
         .replace('None', np.nan)\
         .replace('NaN', np.nan)\
         .replace('null', np.nan)\
         .replace('Null', np.nan)\
         .replace('NULL', np.nan)\
         .replace('', np.nan)

    # 先把已知是缺失的值转化成NA，这样可以确保其不会干扰别的值得缺失赋值
    if len(known_missing) > 0:
        known_missing_values = list(known_missing.keys())
        known_missing_values_str = [str(i) for i in known_missing_values]
        known_missing_values = known_missing_values + known_missing_values_str

    unq_data_sources = var_dict['数据源'].unique()
    new_X_list = []
    for data_source in unq_data_sources:
        data_sources_vars = var_dict.loc[var_dict['数据源']==data_source, '指标英文'].unique()
        data_sources_vars = list(set(data_sources_vars).intersection(set(X.columns)))
        if verbose & (len(set(downflagmap.keys()).intersection(set(data_sources_vars))) == 0):
            print("Warnings: downflag variable is not in downflagmap provided for %s" % data_source)
        else:
            the_downflag_var = list(set(downflagmap.keys()).intersection(set(data_sources_vars)))

        downflag_vars_x = [i for i in data_sources_vars if 'downflag' in i.lower()]
        data_sources_vars = [i for i in data_sources_vars if 'downflag' not in i.lower()]
        sub_X = X[data_sources_vars].copy()
        checker = len(data_sources_vars)

        if len(known_missing) > 0:
            missing_bool = (pd.isnull(sub_X) | sub_X.isin(known_missing_values))
        else:
            missing_bool = pd.isnull(sub_X)

        num_missing = missing_bool.sum(1)

        sub_X = sub_X.fillna(-9999)
        sub_x1 = sub_X.loc[num_missing==checker].copy()
        sub_x2 = sub_X.loc[num_missing!=checker].copy()
        sub_x1.replace(-9999, -8888, inplace=True)
        sub_x2.replace(-9999, -8887, inplace=True)
        new_sub_X = pd.concat([sub_x1, sub_x2]).sort_index()

        if len(downflagmap) > 0:
            for downflag_var in the_downflag_var:
                to_replace = downflagmap.get(downflag_var, {})
                if len(to_replace) > 0:
                    for flag_value, replace_value in list(to_replace.items()):
                        new_sub_X.loc[X[downflag_var].isin([flag_value, str(flag_value)]), :] = replace_value

        new_sub_X = new_sub_X.merge(X[downflag_vars_x], left_index=True, right_index=True)
        new_X_list.append(new_sub_X)

    new_X = pd.concat(new_X_list, axis=1)


    if len(known_missing) > 0:
        for known_missing_value, replace_value in list(known_missing.items()):
            new_X = new_X.replace(known_missing_value, replace_value)\
                         .replace(str(known_missing_value), replace_value)

    return new_X

def encode_categorical_by_all(X,var_dict):
    """
    对数据集中的类别型变量进行类型转换
    
    -----------------------------
    
    Args:
        X(pd.DataFrame): 数据集
        var_dict(pd.DataFrame): 变量字典
    
    Return:
        x(dataFrame): 编码后的数据集
        all_encoding_map(dict): 编码映射表
    """
    if X.isnull().sum().sum() > 0:
        print('数据集中存在缺失值, 请先进行缺失值处理')
        return 
    
    x = copy.deepcopy(X)
    
    cat_vars = var_dict[var_dict['数据类型'] == 'varchar']['指标英文']
    all_encoding_map = {}
    if len(cat_vars) > 0:
        le = preprocessing.LabelEncoder()
        for var in cat_vars:
            encoded_col = le.fit_transform(x[var])
            x.loc[:, var] = pd.Series(encoded_col, index=x.index, name=var)
#            all_encoding_map[var] = dict(zip(le.classes_, encoded_col))
            all_encoding_map[var] = dict(zip(range(len(le.classes_)),le.classes_)) 
        print("字符型变量编码结束")
    else:
        print("数据集中无字符型变量")
        
    return x, all_encoding_map

def is_numeric(x):
    """
    Args:
        x(任意数据类型)
    Returns:
        True for numeric; Flase for other datatypes
    """
    try:
        if isinstance(x, pd.Series):
            x.astype('float')
        else:
            float(x)
    except ValueError:
        return False
    except TypeError:
        return False
    else:
        return True
    
def optimal_mem(X, var_dict, verbose=False):
    """
    通过对数据类型的自动检测优化数据集, 提升训练速度, 同时将数据类型更新到变量字典中.
    
    --------------------------------------------------------------------
    
    Args:
        X(pd.DataFrame): 数据集, 建议在处理完缺失值进行数据类型的优化
        var_dict(pd.DataFrame): 变量字典, 必要列为['指标英文', '数据源', '指标中文', '指标类型']
        
    Returns:
        x_(pd.DataFrame): 优化后的数据集
        var_dict_(pd.DataFrame): 更新后的变量字典, 新增一列 '数据类型'
    """
    
    if X.isnull().sum().sum() > 0:
        print('数据集中存在缺失值, 请先进行缺失值处理')
        return 
    
    x_ = copy.deepcopy(X)
    var_dict_ = copy.deepcopy(var_dict)
    var_dict_ = var_dict_.set_index('指标英文')
    var_dict_['数据类型'] = np.nan
    
    numerics = ['int', 'int8','int16', 'int32', 'int64', 'float', 'float16', 'float32', 'float64']
    start_mem = x_.memory_usage().sum() / 1024**2  
    for col in x_.columns:
        col_type = x_[col].dtypes
        if verbose:
            print(col, col_type)
            
        if not is_numeric(x_[col]):
            col_type = 'str'
            var_dict_.loc[col, '数据类型'] = 'varchar'
        else:
            if col_type not in numerics:
                col_type = 'float'
                
        if col_type in numerics:
            c_min = float(x_[col].min())
            c_max = float(x_[col].max())
            if str(col_type)[:5] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    x_[col] = x_[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    x_[col] = x_[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    x_[col] = x_[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    x_[col] = x_[col].astype(np.int64) 
                var_dict_.loc[col, '数据类型'] = 'integer'
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    x_[col] = x_[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    x_[col] = x_[col].astype(np.float32)
                else:
                    x_[col] = x_[col].astype(np.float64)  
                var_dict_.loc[col, '数据类型'] = 'float'
            
    end_mem = x_.memory_usage().sum() / 1024**2
    if verbose:
        print('数据集原始大小{:.2f}'.format(start_mem))
        print('数据集优化后大小{:5.2f} Mb, 压缩了({:.1f}%)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        
    var_dict_.reset_index(inplace=True)
    return x_, var_dict_


def is_numeric(x):
    """
    检测输入是否为数字, 是返回True; 反之返回False
    --------------------------------------------------------------------
    Args:
        x(任意数据类型)
    Returns:
        True for numeric; Flase for other datatypes
    """
    try:
        if isinstance(x, pd.Series):
            x.astype('float')
        else:
            float(x)
    except ValueError:
        return False
    except TypeError:
        return False
    else:
        return True

def crosstab_c(x, y):
    test = pd.crosstab(x.astype(str), y, margins = True,dropna=False)
    test['badrate'] = test[1.0]/test['All']
    return test


def recur_dictify(frame):
    """
    通过recursion的方法把评分卡的转化成可以部署在线上的使用的nested dictionary

    Args:
    frame (pd.DataFrame): 第一列必须是指标英文名，第二列必须是分箱名称，第3列是分数。

    Returns:
    d (dict): nested dictionary
    {"var_name": {
        'bin_name': -19,
        'bin_name2': 19
        }
    }
    """
    if len(frame.columns) == 1:
        if frame.values.size == 1:
            return frame.values[0][0]
        return frame.values.squeeze()
    grouped = frame.groupby(frame.columns[0])
    d = {k: recur_dictify(g.iloc[:,1:]) for k,g in grouped}
    return d



def process_rebin_spec(rebin_spec, var_dict, selected, production_map={}):
    """
    将rebin_spec整理成可以用于部署的格式

    Args:
    rebin_spec (dict): 用于分箱的dict
    var_dict (pd.DataFrame): 标准变量字典
    selected (list): 决定部署的最终变量名列表
    production_map (dict): 如果建模名称和生产部署的名称不一致，则传入这一dict。key为建模时
        使用的变量英文名，value为生产部署的变量英文名

    Returns:
    new_dict (dict): 用于部署的分箱dict
    """
    var_type = var_dict.loc[var_dict['指标英文'].isin(selected), ['指标英文', '数据类型']]
    numerical = var_type.loc[var_type['数据类型']!='varchar', '指标英文'].tolist()
    categorical = var_type.loc[var_type['数据类型']=='varchar', '指标英文'].tolist()
    new_dict = {}
    d1 = {production_map[k]:v for k,v in list(rebin_spec.items()) if k in numerical and k in production_map}
    d2 = {k:v for k,v in list(rebin_spec.items()) if k in numerical and k not in production_map}
    new_dict['numerical'] = dict(d1, **d2)
    # categorical的mapping格式需要换成蒋宏的那种
    tmp_dict = {}
    for col, spec in list(rebin_spec.items()):
        if col in categorical:
            if col in production_map:
                the_key = production_map[col]
            else:
                the_key = col
            tmp_dict[the_key] = {}
            for label, val_list in list(spec.items()):
                for val in val_list:
                    tmp_dict[the_key][val] = label

    new_dict['categorical'] = tmp_dict

    return new_dict



def process_bin_to_score(score_card):
    """
    将score_card整理成可以用于部署的格式

    Args:
    score_card (pd.DataFrame): 部署excel表里面的评分卡表。须包含 u'中间层指标名称', u'分箱',
        u'变量打分'这三个列名。请传入生产部署评分卡文件

    Returns:
    new_dict (dict): 用于部署的分箱dict
    """
    rename_map = {}
    for i in score_card.columns:
        if i == '打分':
            rename_map[i] = '变量打分'
        if i == '变量英文':
            rename_map[i] = '中间层指标名称'
        if i == '指标英文':
            rename_map[i] = '中间层指标名称'
        if i == '指标中文':
            rename_map[i] = '变量中文'

    if len(rename_map) > 0:
        score_card = score_card.rename(columns=rename_map)

    score_card2 = score_card[['中间层指标名称', '分箱', '变量打分']]
    bin_to_score = recur_dictify(score_card2)
    for var_name, var_score_dict in bin_to_score.items():
        for bin_name, bin_score in var_score_dict.items():
            bin_to_score[var_name][bin_name] = int(bin_score)

    intercept_col = 'const'
    if 'intercept' in list(bin_to_score.keys()):
        del bin_to_score['intercept']
        intercept_col = 'intercept'
    try:
        bin_to_score['const'] = int(score_card.loc[score_card.loc[:, '中间层指标名称']==intercept_col, '变量打分'].iloc[0])
    except:
        bin_to_score['const'] = int(score_card.loc[score_card.loc[:, '变量中文']=='截距分', '变量打分'].iloc[0])


    return bin_to_score


def auto_deplyment_testcases(score_card_prod, model_name):
    """
    用production的score_card生成测试用例

    Args:
    score_card_prod (pd.DataFrame): 生产部署评分卡文件。
    model_name (str): 部署时传送给开发用于调用的modelName

    Returns:
    test_cases (pd.DataFrame): 返回测试用例字段
    """
    score_card_prod = score_card_prod.loc[score_card_prod['分箱'].notnull()]
    SCRIPT_TEMPLATE = """
    {"modelName":"%s","productName":"%s","applyId": 123, "%s":"%s"}
    """

    def _get_test_var_val(one_row):
        if '指标分类' in one_row.index:
            data_source = one_row['指标分类']
        else:
            data_source = one_row['数据源']

        if data_source != '同盾联合建模':
            if one_row['数据类型'] == 'varchar':
                    contents = json.loads('["' + one_row['分箱'].replace(', ', '", "') + '"]')
                    r_list = []
                    for var_val in contents:
                        tmp = one_row.copy()
                        tmp.loc['mock_var_val'] = var_val
                        r_list.append(tmp.to_frame().transpose())
                    result = pd.concat(r_list)
            else:
                tmp = one_row['分箱'].replace('(', '').replace(']', '')\
                                    .split(', ')
                if tmp[0] == '-inf':
                    left_bound = -100000
                else:
                    left_bound = float(tmp[0])
                right_bound = tmp[-1]
                if len(tmp)>1 and one_row['数据类型'] == 'integer':
                    left_bound += 1
                elif len(tmp)>1:
                    left_bound += 0.00001
                row1 = one_row.copy()
                row2 = one_row.copy()
                row1.loc['mock_var_val'] = left_bound
                row2.loc['mock_var_val'] = right_bound
                result = pd.concat([row1.to_frame().transpose(), row2.to_frame().transpose()])
            result.loc[:, 'var_param'] = result.apply(lambda row: SCRIPT_TEMPLATE % (model_name, model_name, row['中间层指标名称'], row['mock_var_val']), axis=1)
            return result

    r_list = []
    for index, row in score_card_prod.iterrows():
        rr = _get_test_var_val(row)
        r_list.append(rr)
    final_result = pd.concat(r_list)
    print(final_result.columns.values)
    rename_map = {'中间层指标名称':'middler_varname','分箱':'var_bin','打分':'bin_score'}
    final_result.rename(columns=rename_map, inplace=True)
    final_result = final_result.loc[final_result.mock_var_val!='inf'].drop_duplicates()
    keep_cols = ['middler_varname','输出打分指标名称','var_bin','bin_score','mock_var_val','var_param']
    return final_result[keep_cols]


def produce_http_test_case(score_card_prod, model_name, product_name, local=False):
    """
    用production的score_card生成测试用例

    Args:
    score_card_prod (pd.DataFrame): 生产部署评分卡文件。
    model_name (str): 部署时传送给开发用于调用的modelName
    product_name (str): product_name
    local: 是否是本地环境。default=False.为线上rundeck发布的QA环境

    Returns:
    test_cases (pd.DataFrame): 返回测试用例字段
    """
    score_card_prod = score_card_prod.loc[score_card_prod['分箱'].notnull()]
    if local:
        SCRIPT_TEMPLATE = """
        curl -d 'varparams={"modelName":"%s", "productName": "%s",
        "applyId": 123, "%s":"%s"}'
        -X POST http://localhost:8080/modelInvoke
        """
    else:
        SCRIPT_TEMPLATE = """
        curl -d 'varparams={"modelName":"%s", "productName": "%s",
        "applyId": 123, "%s":"%s"}'
        -X POST http://dnginj-ml-qa-01.idumiao.com/modelInvoke
        """

    def _get_test_var_val(one_row):
        if '指标分类' in one_row.index:
            data_source = one_row['指标分类']
        else:
            data_source = one_row['数据源']

        if data_source != '同盾联合建模':
            if one_row['数据类型'] == 'varchar':
                    contents = json.loads('["' + one_row['分箱'].replace(', ', '", "') + '"]')
                    r_list = []
                    for var_val in contents:
                        tmp = one_row.copy()
                        tmp.loc['var_val'] = var_val
                        r_list.append(tmp.to_frame().transpose())
                    result = pd.concat(r_list)
            else:
                tmp = one_row['分箱'].replace('(', '').replace(']', '')\
                                    .split(', ')
                if tmp[0] == '-inf':
                    left_bound = -100000
                else:
                    left_bound = float(tmp[0])
                right_bound = tmp[-1]
                if len(tmp)>1 and one_row['数据类型'] == 'integer':
                    left_bound += 1
                elif len(tmp)>1:
                    left_bound += 0.00001
                row1 = one_row.copy()
                row2 = one_row.copy()
                row1.loc['var_val'] = left_bound
                row2.loc['var_val'] = right_bound
                result = pd.concat([row1.to_frame().transpose(), row2.to_frame().transpose()])
            result.loc[:, 'curl_script'] = result.apply(lambda row: SCRIPT_TEMPLATE % (model_name, product_name, row['中间层指标名称'], row['var_val']), 1)
            return result

    r_list = []
    for index, row in score_card_prod.iterrows():
        rr = _get_test_var_val(row)
        r_list.append(rr)

    final_result = pd.concat(r_list)
    return final_result.loc[final_result.var_val!='inf'].drop_duplicates()



def crosstab_percent(row_data, col_data, data_format='wide'):
    """
    将2个数据crosstab并计算col_data各组内的row_data各分箱的占比。

    Args:
    row_data: binned data
    col_data: binned data
    data_format (str): ['wide', 'long']. If data_format='wide', will return wide
        format data. If data_format='long', will return long format data. 一列列\
        名为row_data中的列名，一列为col_data中的列名，还有一列列名为'value'

    Return:
    plot_data: format based on the data_format argument passed in
    """
    if type(row_data) == pd.DataFrame:
        x_name = row_data.iloc[:, 0].name
    elif type(row_data) == pd.Series:
        x_name = row_data.name
    cross_count = pd.crosstab(row_data.astype(str), col_data.astype(str))
    # 计算每个group的分布占比
    cross_pct = cross_count * 1.0 / cross_count.sum()

    if data_format == 'long':
        cross_pct = cross_pct.reset_index()
        plot_data = pd.melt(cross_pct, id_vars=x_name, \
                        value_vars=[i for i in cross_pct.columns if i!= x_name])
        return plot_data

    if data_format == 'wide':
        return cross_pct




idcard_province_map = {
    "11": "北京市",
    "12": "天津市",
    "13": "河北省",
    "14": "山西省",
    "15": "内蒙古",
    "21": "辽宁省",
    "22": "吉林省",
    "23": "黑龙江省",
    "31": "上海市",
    "32": "江苏省",
    "33": "浙江省",
    "34": "安徽省",
    "35": "福建省",
    "36": "江西省",
    "37": "山东省",
    "41": "河南省",
    "42": "湖北省",
    "43": "湖南省",
    "44": "广东省",
    "45": "广西",
    "46": "海南省",
    "50": "重庆市",
    "51": "四川省",
    "52": "贵州省",
    "53": "云南省",
    "54": "西藏",
    "61": "陕西省",
    "62": "甘肃省",
    "63": "青海省",
    "64": "宁夏",
    "65": "新疆",
    "71": "台湾省",
    "81": "香港",
    "82": "澳门",
}

def get_province_from_idcardno(idcardno):
    first2digit = str(idcardno)[:2]
    return idcard_province_map.get(first2digit, '-8887')

def convert_rebin_spec2XGB_rebin_spec(rebin_spec):
    """
    将自动分箱的rebin_spec文件转换为xgboost使用的格式

    Args:
    rebin_spec (dict): 分箱文件

    Returns:
    rebin_spec (dict): xgboost分箱文件
    """
    for i in rebin_spec.keys():
        try:
            rebin_spec[i]['cut_boundaries'] = sorted(set(rebin_spec[i]['cut_boundaries'] \
                                                         + rebin_spec[i]['other_categories'] \
                                                         + [-np.inf, np.inf]))
            rebin_spec[i]['other_categories'] = []
        except:
            pass

    return rebin_spec

# 该方法已废弃，建议不要再用
def xgb_dummy_var_name_map_manipulate(X_train_xgboost,X_test_xgboost,dummy_var_name_map):
    """
    将dummy_var_name_map从新编辑（中间层无法识别特殊字符），并且修改train,test中相应的字段。

    Args:
    X_train_xgboost: 训练集数据
    X_test_xgboost: 测试集数据
    dummy_var_name_map (dict): dummy变量对应的字典

    Return:
    dummy_var_name_map_final(dict): 编辑后dummy变量对应的字典
    """
    dummy_var_name_map_list = list(dummy_var_name_map.keys())
    dummy_var_name_map_v2 = {}
    for i in range(len(dummy_var_name_map_list)):
        dummy_var_name_map_v2[dummy_var_name_map_list[i]] = dummy_var_name_map_list[i].split('DUMMY')[0]+'DUMMY%s'%str(i)
    X_train_xgboost.rename(columns=dummy_var_name_map_v2,inplace=True)
    X_test_xgboost.rename(columns=dummy_var_name_map_v2,inplace=True)
    dummy_var_name_map_final = {}
    for i in dummy_var_name_map_list:
        dummy_var_name_map_final[dummy_var_name_map_v2[i]] = dummy_var_name_map[i]
    return dummy_var_name_map_final