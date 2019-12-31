# encoding=utf8
"""
将numerical数据自动分箱，算WOE，IV，score distribution等metrics

Owner: 胡湘雨

最近更新：2017-12-07
"""
import os
import json
import logging
from itertools import chain
from math import log
try:
    import xgboost as xgb
except:
    pass
import pandas as pd
import numpy as np
from jinja2 import Template
from sklearn import preprocessing, tree, metrics
from sklearn.linear_model import Lasso, LassoCV
from scipy.stats import stats
from copy import deepcopy

from utils3.data_io_utils import *
from utils3.misc_utils import *
from functools import reduce
from math import log

"""
以下为rebin_spec的格式样例。rebin_spec是用来将变量分箱的画界定义
"""

# rebin_spec = {
#     u'c_province': {
#             'loc_1': [u'海南省', u'北京市', u'上海市', u'浙江省'],
#             'loc_2': [u'湖南省', u'安徽省', u'广西', u'江西省', u'江苏省'],
#             'loc_3': [u'云南省', u'贵州省', u'甘肃省', u'广东省', u'黑龙江省', u'四川省', u'天津市'],
#             'loc_4': [u'陕西省', u'重庆市', u'湖北省', u'河北省', u'辽宁省'],
#             'loc_5': [u'吉林省', u'福建省', u'山西省', u'河南省', u'山东省'],
#         },
#     u'n_age': [19.9, 25, 35, 45, 55],
#     u'n_avgMonthCallTime': [-np.inf, 60, 500, 900, 2000, np.inf],
#     u'n_cardTerm': [0, 24, 36, 48, 60, np.inf],
#     u'n_consumeTop': [0, 50, 80, 100],
#     u'n_creditWD3Months': {
#             '0': [0],
#             '[1,2,3]': [1,2,3],
#             '[4,5]': [4,5],
#             '6': [6]
#         },
#     u'n_currentJobyear': [-np.inf, 1, 2, 3],
#     u'n_goOut120': [-np.inf, 0, np.inf],
#     u'n_ivsScore': [-np.inf, 77, 86, 94, 100],
#     u'n_mealsNum': [-np.inf, 1, 3, np.inf],
#     u'n_networkTime6': [0, 1369, 1673, np.inf],
#     u'n_zhimaScore': [-np.inf, 580, 617, 632, 637, 648, 662, 678, 694, 715, np.inf],
#     # u'n_tongdunIdMultiLoanNumPf': [-np.inf, 8, 15, np.inf],
# }




class BinWoe(object):
    def __init__(self):
        pass
    
    def add_origin_categorical_val(self, woe_iv_df, all_encoding_map):
        for k, v in all_encoding_map.items():
            woe_iv_df.loc[woe_iv_df['指标英文'] == k, '分箱对应原始分类'] = woe_iv_df[woe_iv_df['指标英文'] == k]['分箱'].astype(int).map(v)


    def categorical_cut_bin(self, x, y, min_size=None, num_max_bins=10):
        """
        单变量：用decision tree给分类变量分箱

        Args:
        x (pd.Series): the original cateogorical variable x，process_missing()处理过的
        y（pd.Series): label
        min_size (int): the minimum size of each bin. If not provided, will be set
        as the max of 200 or 3% of sample size
        num_max_bins (int): the max number of bins to device. default=6

        Returns:
        new_x (pd.Series): the new x with new cateogory labeled with numbers
        encode_mapping (dict): the mapping dictionary between original x and label
            encoded x used for tree building
        spec_dict (dict): the mapping dictionary of new categorical label and original
            categories. Format as rebin_spec as shown above
        """
        if not min_size:
            min_size = max(200, int(len(x)*0.03))

        x_encoded = label_encode(x)
        # label encoded mapping to original x
        map_df = pd.DataFrame({'original': x, 'encoded': x_encoded}).drop_duplicates()
        encode_mapping = dict(list(zip(map_df.original, map_df.encoded)))

        # 因为之后missing的这一个level会被单独拎出来。要确保拎出来之后的level依然有至少3%或50条数据
        num_missing1 = 0
        num_missing2 = 0
        num_missing3 = 0

        if -9999 in x.tolist() or '-9999' in x.tolist():
            num_missing1 = sum(x==-9999) + sum(x=='-9999')
        if -8888 in x.tolist() or '-8888' in x.tolist():
            num_missing2 = sum(x==-8888) + sum(x=='-8888')
        if -8887 in x.tolist() or '-8887' in x.tolist():
            num_missing3 = sum(x==-8887) + sum(x=='-8887')

        min_samples_leaf = min_size + num_missing1 + num_missing2 + num_missing3
        tree_obj = tree.DecisionTreeClassifier(criterion ='entropy', min_samples_leaf=min_samples_leaf,\
                                               max_leaf_nodes=num_max_bins)
        tree_obj.fit(x_encoded.values.reshape(-1, 1), y)
        # -2 is TREE_UNDEFINED default value
        cut_point = tree_obj.tree_.threshold[tree_obj.tree_.threshold != -2]
        #print cut_point
        new_x = pd.Series(np.zeros(x.shape), index=x.index)
        if len(cut_point) > 0:
            cut_point.sort()
            new_x.loc[x_encoded < cut_point[0]] = 1
            for i in range(len(cut_point)-1):
                new_x[(x_encoded>=cut_point[i]) & (x_encoded<cut_point[i+1])] = i+2
            new_x[x_encoded>=cut_point[len(cut_point)-1]] = len(cut_point) + 1
        else:
            new_x = pd.Series(np.ones(x_encoded.shape), index=x.index)

        if -9999 in x.tolist() or '-9999' in x.tolist():
            new_x.loc[x.isin([-9999, '-9999'])] = '-9999'
        if -8888 in x.tolist() or '-8888' in x.tolist():
            new_x.loc[x.isin([-8888, '-8888'])] = '-8888'
        if -8887 in x.tolist() or '-8887' in x.tolist():
            new_x.loc[x.isin([-8887, '-8887'])] = '-8887'

        df1 = x.to_frame('original_x')
        df2 = new_x.to_frame('new_x_category')
        df3 = pd.concat([df1,df2], axis=1)
        df4 = df3.drop_duplicates()
        df4 = df4.loc[(~df4.new_x_category.isin([-8888, -8887, -9999, '-8888', '-8887', '-9999']))]
        spec_dict = {}
        for index, row in df4.iterrows():
            if row['new_x_category'] not in spec_dict:
                spec_dict[row['new_x_category']] = []
            spec_dict[row['new_x_category']].append(row['original_x'])

        return new_x, encode_mapping, spec_dict


    def numerical_cut_bin(self, x, var_type, num_max_bins=10, min_size=None, missing_alone=True):
        """
        Auto classing for one numerical variable. If x has more than 100 unique
        values, then divide it into 20 bins, else if x has more than 10, then divide
        it into 10 bins. else if x has more than 3 unique values, divide accordingly,
        else, keep it as it is and convert to str.
        All 0 and -8888, -8887, -9999 自己持有一箱。当数量unique值<=10且>3时，则<=1
        & > -8887的数值每一个值是一箱

        Args:
        x (pd.Series): original x values, numerical values. process_missing()处理过的
        var_type (str): ['integer', 'float']
        num_max_bins (int): number of max bins to cut. default = 10
        min_size (int): the minimum size of each bin.If not provided, will be set
        as the max of 200 or 3% of sample size
        missing_alone (bool): default=True. -9999, -8888, -8887各自单独一箱.
            If false，缺失值被当成是正常的数据数值参与分箱，可能和实际值的最低箱分在一起

        Returns:
        x_category (pd.Series): binned x
        """
        num_max_bins = int(num_max_bins)
        if num_max_bins <= 2 or pd.isnull(num_max_bins):
            raise "num_max_bins is too small or None. value is %s" % num_max_bins

        if x.dtypes == 'O':
            x = x.astype(float)
        if not min_size:
            min_size = max(200, int(len(x)*0.03))

        precision = 0
        if var_type == 'float':
            if max(x) <= 1:
                precision = 4
            elif max(x) <= 10:
                precision = 3


        if x.nunique() > 3:
            if (-8888 in x.tolist() or -8887 in x.tolist() or -9999 in x.tolist()) & missing_alone:
                x_missing1 = x[x.isin([-8888, -8887, -9999])].copy().astype('str')
                x_not_missing = x[(~x.isin([-8888, -8887, -9999]))].copy()
                quantiles_list = np.linspace(0, 1, num=num_max_bins)
                bounds = np.unique(x_not_missing.quantile(quantiles_list).round(precision).tolist())[1:-1]#去掉最小和最大
                min_value = min(x_not_missing)
                if min_value > -8887:
                    bounds = np.unique([-8887] + list(bounds) + [np.inf])
                else:
                    bounds = np.unique([-np.inf] + list(bounds) + [np.inf])
                x_not_missing_binned = pd.cut(x_not_missing, bounds)
                x_category = pd.concat([x_missing1, x_not_missing_binned]).sort_index().astype('str')
            else:
                quantiles_list = np.linspace(0, 1, num=num_max_bins)
                bounds = np.unique(x.quantile(quantiles_list).round(3).tolist())[1:-1]
                bounds = [-np.inf] + list(bounds) + [np.inf]
                x_category = pd.cut(x, bounds).astype('str')
        else:
            x_category = x.copy().astype('str')

        return x_category



    def binning(self, X, y, var_dict, num_max_bins=10, verbose=True, min_size=None, missing_alone=True):
        """
        Auto Classing
        如果分类变量类别<=5,则保留原始数据

        Args:
        X (pd.Series): 变量宽表，原始值的X
        y (pd.Series): label
        var_dict (pd.DataFrame): 标准变量字典表
        num_max_bins (int): 分类变量auto classing的分箱数
        verbose (bool): default=True. If set True, will print process logging
        min_size (int): the minimum size of each bin.If not provided, will be set
        as the max of 200 or 3% of sample size
        missing_alone (bool): 用于self.numerical_cut_bin()。default=True. -9999, -8888, -8887各自单独一箱.
            If false，缺失值被当成是正常的数据数值参与分箱，可能和实际值的最低箱分在一起

        Returns：
        X_cat (pd.Series): binned X
        all_encoding_map (dict): encoding map for all variables, keys are variable names
                类别变量原始值对应分箱值
                 {'client': {'Android': 0,
                  'DmAppAndroid': 1,
                  'DmAppIOS': 2,
                  'IOS': 3,
                  'JmAppAndroid': 4,
                  'JmAppIOS': 5,
                  'Touch': 6,
                  'WeChat': 7,
                  'Web': 8}}
        all_spec_dict (dict): rebin spec dictionary for all variables, keys are variable names
                类别变量分箱值对应的原始值列表
                {'client': {1.0: ['Android', 'DmAppAndroid', 'DmAppIOS'],
                  2.0: ['IOS', 'JmAppAndroid', 'JmAppIOS'],
                  3.0: ['WeChat', 'Touch'],
                  4.0: ['Web']}}
        """
        X_cat = X.copy()
        all_encoding_map = {}
        all_spec_dict = {}
        for col in X_cat.columns:
            if verbose:
                logging.log(18, col + ' starts binning')

            # check data type
            if '数据类型' in var_dict.columns.values:
                var_type = str(var_dict.loc[var_dict['指标英文']==col, '数据类型'].iloc[0])
            else:
                if X[col].dtype == 'object':
                    var_type = 'varchar'
                else:
                    var_type = 'float'
              
            # cut bins for varchar and float variables, 将分箱后的label显示出来
            if var_type in ['integer', 'float']:
                X_cat.loc[:, col] = self.numerical_cut_bin(X[col], var_type, num_max_bins=num_max_bins, min_size=min_size, missing_alone=missing_alone)
            elif var_type == 'varchar':
                if X[col].nunique() > 5:
                    X_cat.loc[:, col], encode_mapping, spec_dict = self.categorical_cut_bin(X[col].astype(str), y, min_size=min_size, num_max_bins=num_max_bins)
                    all_encoding_map[col] = encode_mapping
                    all_spec_dict[col] = spec_dict

        return X_cat, all_encoding_map, all_spec_dict


    def numeric_monotonic_binning(self, x, y, n):
        """
        单个变量Automatically bin the continuous variable that has linear WOE

        Args:
        x (pd.Series): original data of one variable
        y (pd.Series): y label

        Returns:
        binned_x (pd.Series): of len(x), NA values are grouped into 'missing' bin
        temp_woe (pd.DataFrame): results return by calculate_woe()
        """
        # fill missings with median
        x_complete = x[~x.isin([-9999, -8888])].copy()
        x_missing = x[x.isin([-9999, -8888])].copy()
        r = 0
        while np.abs(round(r, 3)) != 1:
            binned_x = pd.qcut(x_complete, n, duplicates='drop').astype('str')
            if not x_missing.empty:
                # x原有的index还有，通过sort他来保证和Y的对齐
                binned_x = pd.concat([binned_x, x_missing]).sort_index()
            temp_woe = self.calculate_woe(binned_x, y)
            d1 = pd.DataFrame({"x": x, "bin": binned_x})
            bin_x_mean = d1.groupby('bin', as_index = False).x.mean()
            d2 = bin_x_mean.merge(temp_woe[['bin', 'WOE']])
            if -8888 in d2.bin.tolist() or -9999 in d2.bin.tolist():
                d2 = d2.loc[~d2.bin.isin([-8888, -9999]), :]
            r, p = stats.spearmanr(d2.x, d2.WOE)
            if n > 2:
                n = n - 1
            else:
                break

        return binned_x, temp_woe


    def monotonic_binning(self, X, y, n=20):
        """
        整个变量matrix： Automatically bin the continuous variable that has
        linear WOE。 内部套用了numeric_monotonic_binning(x, y, n)

        Args:
        X (pd.Series): original data of one variable
        y (pd.Series): y label
        n (int): default = 20

        Returns:
        binned_X (pd.DataFrame): of len(x), NA values are grouped into 'missing' bin
        woe_iv_df (pd.DataFrame): results return by calculate_woe()
        """
        new_X_list = []
        woe_list = []
        numerical_vars = [i for i in X.columns if i[:2] == 'n_']
        categorical_vars = [i for i in X.columns if i[:2] == 'c_']
        other = [i for i in X.columns if i not in numerical_vars and i not in categorical_vars]
        for col in numerical_vars:
            logging.log(18, col + ' starts monotonic binning')
            binned_x, var_woe = numeric_monotonic_binning(X[col], y, n)
            new_X_list.append(binned_x)
            woe_list.append(var_woe)

        for col in categorical_vars+other:
            new_x = X[col].copy()
            new_x.loc[pd.isnull(new_x)] = 'missing'
            new_X_list.append(new_x.astype('str'))
            var_woe = self.calculate_woe(new_x, y)
            woe_list.append(var_woe)

        binned_X = pd.concat(new_X_list, 1)
        woe_iv_df = pd.concat(woe_list, ignore_index=True)
        return binned_X, woe_iv_df



    def obtain_boundaries(self, var_bin, missing_alone=True):
        """
        根据已分好箱的变量值，提取分箱界限值.主要应用于连续变量自动分箱的情况下，需要从分好箱
        的数据中提取分箱界限

        Args:
        var_bin (pd.Series): 单变量，已经binning好了的x
        missing_alone (bool): default=True 将min_bound和0作为分箱边界。If False不单独处理，
            仅将最小值和最大值替换为inf

        Returns:
        result (dict): 分箱界限stored in a dict
        {
            'other_categories': [-8888.0, 0.0],
            'cut_boundaries': [0.0, 550.0, 2110.0, 5191.0, 9800.0, 16000.0, 28049.0, 54700.0, inf]
        }
        """
        unique_bin_values = var_bin.astype('str').unique()
        boundaries = [i.replace('(', '').replace(']', '').replace('[', '').replace(')', '').split(', ') for i in unique_bin_values if ',' in i]
        if boundaries:
            boundaries = [float(i) for i in list(chain.from_iterable(boundaries)) if i not in ['nan', 'missing']]
            min_bound = np.min(boundaries)
            max_bound = np.max(boundaries)
            boundaries = [i for i in boundaries if i != min_bound and i != max_bound]
            if missing_alone:
                if ('0' in unique_bin_values or '0.0' in unique_bin_values) and min_bound == 0:
                    boundaries.extend([np.inf, min_bound, -np.inf])#新数据集中可能出现小于min_bound的数据
                elif min_bound == -8887:
                    boundaries.extend([np.inf, min_bound, -np.inf])
                else:
                    boundaries.extend([np.inf, -np.inf])
            else:
                boundaries.extend([np.inf, -np.inf])
            boundaries = sorted(set(boundaries))

        not_bin_cat = [float(i) for i in unique_bin_values if ',' not in i]
        result = {
            'cut_boundaries': boundaries,
            'other_categories': not_bin_cat
        }

        return result


    def write_rebin_spec(self, rebin_spec, woe_iv_df, data_path, file_name_label):
        """
        把 numeric_monotonic_binning()弄出来的bin加入rebin_spec里面，以方便之后用rebin_spec
        分train和test data. rebin_spec会save成json

        Args:
        rebin_spec (dict): 如上所示的dict，一般是categorical或是需要手动调整的numerical的bin
        woe_iv_df (pd.DataFrame): 已经选中的变量，且numerical的variable已经被
            numeric_monotonic_binning() 分箱好了

        Returns:
        rebin_spec (dict): updated rebin_spec
        And save
        """
        variables = woe_iv_df.var_code.unique()
        for col in variables:
            if col not in list(rebin_spec.keys()) and col[:2] != 'c_':
                var_bin = woe_iv_df.loc[woe_iv_df.var_code == col, 'bin']
                rebin_spec[col] = obtain_boundaries(var_bin)

        try:
            with open(os.path.join(data_path, 'rebin_spec_%s.txt' % file_name_label), 'w') as outfile:
                json.dump(rebin_spec, outfile)
        except:
            logging.error("Saving failed")
        return rebin_spec



    def convert_to_category(self, X, var_dict, rebin_spec, verbose=True,
                                replace_value='-9999'):
        """
        将原始数值的变量宽表按照rebin_spec进行分箱， rebin_spec格式如上所示。

        Args:
        X (pd.DataFrame): X原始值宽表, process_missing()处理过的
        var_dict(pd.DataFrame): 标准变量字典
        rebin_spec(dict): 定义分箱界限的dict
        verbose (bool): default=True. If set True, will print the process logging

        Returns:
        X_cat (pd.DataFrame): X after binning
        """
        X_cat = X.copy()
        for col, strategy in list(rebin_spec.items()):
            if verbose:
                logging.log(18, col + ' starts checking 数据类型 before the conversion')

            right_type = var_dict.loc[var_dict['指标英文']==col, '数据类型'].iloc[0]

            if col in X_cat.columns:
                if verbose:
                    logging.log(18, col + ' starts the binning conversion')

                if right_type == 'varchar':
                    for category, cat_range in list(strategy.items()):
                        if isinstance(cat_range, list):
                            X_cat.loc[X_cat[col].isin(cat_range), col] = category
                        else:
                            X_cat.loc[(X_cat[col]==cat_range), col] = category
                    #替换后，还有新类别的需要再处理
                    checker_value = list(strategy.keys())+['-9999','-8887','-8888',-9999,-8887,-8888]
                    newobs=~X_cat[col].isin(checker_value)
                    if newobs.sum()>0:
                        if replace_value=='min_value':
                            replace_value = min([str(x) for x in checker_value])
                        logging.warning('{}有新的类别{}出现,替换 {} 条数据值为{}:'.format(col,list(X_cat.loc[newobs, col].unique()),
                                        newobs.sum(),replace_value))
                        X_cat.loc[newobs, col] = replace_value
                else:
                    if X[col].dtypes == 'O':
                        X[col] = X[col].astype(float)

                    if isinstance(strategy, dict) and 'cut_boundaries' in strategy and 'other_categories' in strategy:
                        cut_spec = strategy['cut_boundaries']
                        other_categories = strategy['other_categories']
                        x = X[col].copy()
                        x1 = x.loc[x.isin(other_categories)].copy().astype('str')
                        x2 = x.loc[~x.isin(other_categories)].copy()
                        x2_cutted = pd.cut(x2, cut_spec).astype('str')
                        X_cat.loc[:, col] = pd.concat([x1, x2_cutted]).sort_index()
                    elif isinstance(strategy, list):
                        X_cat.loc[:, col] = pd.cut(X[col], strategy).astype(str)


        return X_cat




    def transform_x_to_woe(self, x, woe):
        """
        Transform a single binned variable to woe value

        Args:
        x (pd.Series): original x that is already converted to categorical, each level should match the name used
            in woe
        woe (pd.Series): contains the categorical level name and the corresponding woe

        Returns:
        x2 (pd.Series): WOE-transformed x

        example:
        >>> x.head()
        0    (-inf, 1.0]
        1    (-inf, 1.0]
        2    (-inf, 1.0]
        3     (3.0, inf]
        4    (-inf, 1.0]
        Name: n_mealsNum, dtype: object
        >>> woe.head()
                   bin       WOE
        0  (-inf, 1.0]  0.027806
        1   (1.0, 3.0] -0.106901
        2   (3.0, inf] -0.179868
        3        -8888  0.443757
        """
        woe.index = woe.bin.astype(str)
        woe_dict = woe.WOE.to_dict()
        x2 = x.copy().astype('str').replace(woe_dict)
        value_type = x2.apply(lambda x: str(type(x)))
        not_converted = value_type.str.contains('str')
        if sum(not_converted) > 0:
            logging.warning("""
            %s 变量包含新值，不在原来的分箱中。
            WOE转换数据为：%s
            未转换成功数据count：%s
            """ % (x.name, json.dumps(woe_dict), x2.loc[not_converted].value_counts()))

            x2.loc[value_type.str.contains('str')] = 0
        return x2.astype(float)


    def transform_x_all(self, X, woe_iv_df):
        """
        Transform binned X to woe value

        Args:
        X (pd.DataFrame): original X that is already converted to categorical,
            each level should match the name used in woe
        woe_iv_df (pd.DataFrame): contains the categorical level name and the corresponding woe. Must
            contain columns: 'bin', 'WOE'. Usually calculate_woe() 返回的result
            should work

        Returns:
        X_woe (pd.DataFrame): WOE-transformed x
        """
        woe_iv_df = woe_iv_df.copy()
        if '指标英文' in woe_iv_df.columns.values:
            woe_iv_df.rename(columns={'指标英文': 'var_code'}, inplace=True)
        if '分箱' in woe_iv_df.columns.values:
            woe_iv_df.rename(columns={'分箱': 'bin'}, inplace=True)

        X_woe = X.copy()
        woe_vars = woe_iv_df.var_code.unique()
        x_vars = X.columns.values
        transform_cols = list(set(woe_vars).intersection(set(x_vars)))
        for var in transform_cols:
            woe = woe_iv_df.loc[woe_iv_df.var_code == var, ['bin', 'WOE']]
            X_woe.loc[:, var] = self.transform_x_to_woe(X_woe[var], woe)
        return X_woe



    def order_bin_output(self, result, col):
        """
        将分箱结果排序，生成一列拍序列。

        Args:
        result (pd.DataFrame): 需要添加排序的数据
        col (str): 分箱的列名
        """
        result = result.copy()
        result.loc[:, col] = result.loc[:, col].astype('str')
        for index, rows in result.iterrows():
            if (',' in rows[col]) & (']' in rows[col] or '(' in rows[col]):
                sort_val = rows[col].replace('(', '')\
                                       .replace(')', '')\
                                       .replace('[', '')\
                                       .replace(']', '')\
                                       .split(', ')[0]
                if 'inf' in sort_val and '-inf' not in sort_val:
                    result.loc[index, 'sort_val'] = 1e10
                elif '-inf' in sort_val:
                    result.loc[index, 'sort_val'] = -1e10
                else:
                    result.loc[index, 'sort_val'] = float(sort_val)
            elif '-8887' in rows[col]:
                result.loc[index, 'sort_val'] = -1e10
            elif '-8888' in rows[col]:
                result.loc[index, 'sort_val'] = -1e11
            elif '-9999' in rows[col]:
                result.loc[index, 'sort_val'] = -1e12
            elif rows[col] == '0' or rows[col] == '0.0' or rows[col] == '0.00':
                result.loc[index, 'sort_val'] = -0.01
            elif rows[col] == '1' or rows[col] == '1.0' or rows[col] == '1.00':
                result.loc[index, 'sort_val'] = 0.9999
            else:
                try:
                    result.loc[index, 'sort_val'] = float(rows[col])
                except:
                    result.loc[index, 'sort_val'] = 0
        result = result.sort_values('sort_val')
        return result

    def calculate_woe(self, x, y,ks_order='eventrate_order'):
        """
        计算某一个变量的WOE和IV

        Args:
        x (pd.Series): 变量列，已经binning好了的
        y (pd.Series): 标签列
        ks_order(str): 选择KS计算的排序方式，默认eventrate_order，可选bin_order

        Returns:
        result (pd.DataFrame): contains the following columns
        ['var_code', 'bin', 'N', 'PercentDist', 'WOE', 'EventRate', 'PercentBad', 'PercentGood', 'IV']
        """
        x = x.astype(str)
        N = x.value_counts().to_frame('N')
        N.loc[:, 'PercentDist'] = (N * 1.0 / N.sum()).N
        dd = pd.crosstab(x, y)
        dd.index.name = None
        dd2 = dd * 1.0 / dd.sum()
        dd2 = dd2.rename(columns={1: 'PercentBad', 0: 'PercentGood'})
        if 1 in dd.columns:
            dd2.loc[:, 'EventRate'] = dd.loc[:, 1] * 1.0 / dd.sum(1)
            dd2.loc[:, 'NBad'] = dd.loc[:, 1]
        else:
            dd2.loc[:, 'EventRate'] = 0
            dd2.loc[:, 'NBad'] = 0
            dd2.loc[:, 'PercentBad'] = 0

        dd2.loc[:, 'PercentBad'] = dd2.PercentBad.round(4)
        dd2.loc[:, 'PercentGood'] = dd2.PercentGood.round(4)

        woe = np.log(dd2['PercentBad'] / dd2['PercentGood'])
        woe.loc[np.isneginf(woe)] = 0
        woe = woe.replace(np.inf, 0).fillna(0)
        iv = sum((dd2.loc[np.isfinite(woe), 'PercentBad'] - dd2.loc[np.isfinite(woe), 'PercentGood']) * woe[np.isfinite(woe)])
        result = dd2.merge(woe.to_frame('WOE'), left_index=True, right_index= True)\
                    .merge(N, left_index=True, right_index= True)
        result.loc[:, 'var_code'] = x.name
        result.loc[:, 'IV'] = iv

        result = result.reset_index()\
                       .rename(columns={'index': 'bin'})

        # 计算KS之前，要按照逾期率从高到低sort
        if ks_order == 'eventrate_order':
            result = result.sort_values('EventRate', ascending=False)
        elif ks_order == 'bin_order':
            result = self.order_bin_output(result,'bin')
        else:
            logging.info("默认排序'eventrate_order'")
            result = result.sort_values('EventRate', ascending=False)

        result.loc[:, 'cumGood'] = result.PercentGood.cumsum()
        result.loc[:, 'cumBad'] = result.PercentBad.cumsum()
        result.loc[:, 'cum_diff'] = abs(result.cumBad - result.cumGood)
        result.loc[:, 'KS'] = max(result.cum_diff)

        # sum all rows
        # add_row = {'var_code': result.var_code.iloc[0],\
        #            'bin': 'ALL', \
        #            'N': result.N.sum(), \
        #            'PercentDist': result.PercentDist.sum(),\
        #            'NBad': result.NBad.sum(),\
        #            'EventRate': result.NBad.sum() * 1.0 / result.N.sum(),
        #            'PercentBad': result.PercentBad.sum(),\
        #            'PercentGood': result.PercentGood.sum()}
        # result = result.append(pd.DataFrame(add_row, [len(result)]))

        # 添加排序列:按照分箱值排序
        result = self.order_bin_output(result, 'bin')

        return result[['var_code', 'bin', 'N', 'PercentDist', 'WOE', 'NBad', \
                       'EventRate', 'PercentBad', 'PercentGood', 'cumBad', \
                       'cumGood', 'cum_diff', 'IV', 'KS', 'sort_val']]


    def calculate_woe_sw(self,x,y,sample_weight,col,ks_order='eventrate_order'):
        """
        计算某一个变量的WOE和IV

        Args:
        x (pd.Series): 变量列，已经binning好了的
        y (pd.Series): 标签列
        sample_weight(pd.Series):样本权重
        col(str):具体变量
        ks_order(str): 选择KS计算的排序方式，默认eventrate_order，可选bin_order

        Returns:
        result (pd.DataFrame): contains the following columns
        ['var_code', 'bin', 'N', 'PercentDist', 'WOE', 'EventRate', 'PercentBad', 'PercentGood', 'IV']
        """
        # 计算分布
        tmp_data=pd.DataFrame([x,y,sample_weight])
        tmp_data=tmp_data.T
        tmp=pd.pivot_table(tmp_data,index=col,values='sample_weight',aggfunc=np.sum)
        tmp['PercentDist']=tmp['sample_weight']/tmp['sample_weight'].sum()
        tmp.rename(columns={'sample_weight':'N'},inplace=True)
        #woe
        woe=pd.pivot_table(tmp_data,index=col,columns='y',values='sample_weight',aggfunc=np.sum).reset_index()
        woe.columns=['bin','NGood','NBad']
        woe.index=woe.bin
        woe['NBad'].fillna(0,inplace=True)
        woe['NGood'].fillna(0,inplace=True)
        woe['PercentBad']=woe['NBad']/woe['NBad'].sum()
        woe['PercentGood']=woe['NGood']/woe['NGood'].sum()
        woe['PercentBad']=woe['PercentBad'].round(4)
        woe['PercentGood']=woe['PercentGood'].round(4)
        woe['PercentBad'].fillna(0,inplace=True)
        woe['PercentGood'].fillna(0,inplace=True)
        woe['EventRate']=woe['NBad']/(woe['NBad']+woe['NGood'])
        woe['WOE']=np.log(woe['PercentBad'] / woe['PercentGood'])
        woe['WOE']=woe['WOE'].apply(lambda x:0 if np.isneginf(x) else x)
        woe['WOE']=woe['WOE'].replace(np.inf,0).fillna(0)
        #计算IV
        woe['IV']=sum((woe.loc[np.isfinite(woe['WOE']), 'PercentBad']-woe.loc[np.isfinite(woe['WOE']), 'PercentGood'])*woe.loc[np.isfinite(woe['WOE']), 'WOE'])
        #合并数据
        result=pd.merge(woe,tmp,left_index=True,right_index=True)
        #计算KS之前，逾期率从高到低sort
        if ks_order=='eventrate_order':
            result=result.sort_values('EventRate',ascending=False)
        elif ks_order=='bin_order':
            result=self.order_bin_output(result,'bin')
        else:
            logging.info("默认排序'eventrate_order'")
            result=result.sort_values('EventRate',ascending=False)
        result.loc[:,'cumGood']=result.PercentGood.cumsum()
        result.loc[:,'cumBad']=result.PercentBad.cumsum()
        result['cum_diff']=abs(result['cumBad']-result['cumGood'])
        result['KS']=max(result['cum_diff'])
        result['var_code']=col
        result.reset_index(drop=True,inplace=True)
        #添加排序列，按照分箱值排序
        result=self.order_bin_output(result,'bin')
        return result[['var_code','bin', 'N', 'PercentDist', 'WOE', 'NBad', \
                       'EventRate', 'PercentBad', 'PercentGood', 'cumBad', \
                       'cumGood', 'cum_diff','IV','KS','sort_val']]

    def calculate_woe_all(self, X, y, var_dict=None, all_spec_dict0=None,
                              verbose=True, ks_order='eventrate_order'):
        """
        计算所有变量的WOE和IV

        Args:
        x (pd.DataFrame): 变量列，已经binning好了的
        y (pd.Series): 标签列
        var_dict (pd.DataFrame): 标准变量字典. 请做一些初期的数据源筛选。因为有些变量在
            不同数据源当中都有，用的也是相同的英文名称
        all_spec_dict (dict): 分类变量的数据在用tree分类后返回的数字标签和相应的原始
            分类对应关系。 categorical_cut_bin()返回的all_spec_dict格式
        verbose (bool): default=True. If set True, will print the process logging
        ks_order: 'eventrate_order' 按照badrate排序后计算ks；'bin_order' 按照变量的分箱顺序计算ks

        Returns:
        woe_iv_df (pd.DataFrame): contains the following columns
        [u'数据源', u'指标英文', u'指标中文', u'数据类型', u'指标类型',\
        u'分箱', 'N', u'分布占比', 'WOE', u'逾期率', u'Bad分布占比',\
        u'Good分布占比', 'IV']
        """
        if all_spec_dict0 is None:
            spec_dict_flag = False
        else:
            spec_dict_flag = True
            all_spec_dict = deepcopy(all_spec_dict0)
            all_spec_dict = {k:v for k,v in list(all_spec_dict.items()) \
                                if var_dict.loc[var_dict['指标英文']==k, '数据类型'].iloc[0]=='varchar'}

        woe_iv_result = []
        for col in X.columns.values:
            if verbose:
                logging.log(18, col + ' woe calculation starts')

            woe_iv_result.append(self.calculate_woe(X[col], y,ks_order=ks_order))

        woe_iv_df = pd.concat(woe_iv_result)
        woe_iv_df.loc[:, 'comment'] = 'useless_0.02_minus'
        woe_iv_df.loc[(woe_iv_df.IV <0.1) & (woe_iv_df.IV >= 0.02), 'comment'] = 'weak_predictor_0.02_0.1'
        woe_iv_df.loc[(woe_iv_df.IV <0.3) & (woe_iv_df.IV >= 0.1), 'comment'] = 'medium_predictor_0.1_0.3'
        woe_iv_df.loc[(woe_iv_df.IV <0.5) & (woe_iv_df.IV >= 0.3), 'comment'] = 'strong_predictor_0.3_0.5'
        woe_iv_df.loc[(woe_iv_df.IV >= 0.5), 'comment'] = 'strong_predictor_0.5_plus'
        woe_iv_df = woe_iv_df.rename(columns = {
                                        'var_code': '指标英文',
                                        'bin': '分箱',
                                        'PercentDist': '分布占比',
                                        'PercentBad': 'Bad分布占比',
                                        'PercentGood': 'Good分布占比',
                                        'EventRate': '逾期率',
                                        'NBad': '坏样本数量',
                                        'cumBad': 'Cumulative Bad Rate',
                                        'cumGood': 'Cumulative Good Rate',
                                        'cum_diff': 'Cumulative Rate Difference'
                                    })
        if var_dict is None:
            var_dict_flag = False
            woe_result = woe_iv_df
        else:
            var_dict_flag = True
            woe_result = var_dict.loc[:, ['数据源', '指标英文', '指标中文', '数据类型', '指标类型']]\
                        .merge(woe_iv_df, on='指标英文', how='right')

        if spec_dict_flag:
            # 加上分类变量的数据在用tree分类后返回的数字标签和相应的原始分类对应关系。
            woe_result.loc[:, '分箱对应原始分类'] = None
            for col, col_spec in list(all_spec_dict.items()):
                for new_label, original_x in list(col_spec.items()):
                    col_spec[str(new_label)] = ', '.join([str(i) for i in original_x])
                woe_result.loc[woe_result['指标英文'] == col, '分箱对应原始分类'] = woe_result.loc[woe_result['指标英文'] == col, '分箱'].astype('str')
                woe_result.loc[woe_result['指标英文'] == col, '分箱对应原始分类'] = woe_result.loc[woe_result['指标英文'] == col, '分箱对应原始分类'].replace(col_spec)
            if var_dict_flag:
                reorder_cols = ['数据源', '指标英文', '指标中文', '数据类型', '指标类型',\
                            '分箱', '分箱对应原始分类', 'N', '分布占比', '坏样本数量', '逾期率', 'WOE', \
                            'Bad分布占比', 'Good分布占比', 'Cumulative Bad Rate',\
                            'Cumulative Good Rate', 'Cumulative Rate Difference',\
                            'IV', 'KS', 'comment', 'sort_val']
            else:
                reorder_cols = ['指标英文','分箱', '分箱对应原始分类', 'N', '分布占比', '坏样本数量', '逾期率', 'WOE', \
                            'Bad分布占比', 'Good分布占比', 'Cumulative Bad Rate',\
                            'Cumulative Good Rate', 'Cumulative Rate Difference',\
                            'IV', 'KS', 'comment', 'sort_val']
        else:
            if var_dict_flag:
                reorder_cols = ['数据源', '指标英文', '指标中文', '数据类型', '指标类型',\
                            '分箱', 'N', '分布占比', '坏样本数量', '逾期率', 'WOE', \
                            'Bad分布占比', 'Good分布占比', 'Cumulative Bad Rate',\
                            'Cumulative Good Rate', 'Cumulative Rate Difference',\
                            'IV', 'KS', 'comment', 'sort_val']
            else:
                reorder_cols = ['指标英文','分箱', 'N', '分布占比', '坏样本数量', '逾期率', 'WOE', \
                            'Bad分布占比', 'Good分布占比', 'Cumulative Bad Rate',\
                            'Cumulative Good Rate', 'Cumulative Rate Difference',\
                            'IV', 'KS', 'comment', 'sort_val']
        result = woe_result[reorder_cols]
        return result

    def calculate_woe_all_sw(self, X, y,sample_weight,var_dict=None, all_spec_dict0=None,
                              verbose=True, ks_order='eventrate_order'):
        """
        计算所有变量的WOE和IV

        Args:
        x (pd.DataFrame): 变量列，已经binning好了的
        y (pd.Series): 标签列
        sample_weight(pd.Series):样本权重
        var_dict (pd.DataFrame): 标准变量字典. 请做一些初期的数据源筛选。因为有些变量在
            不同数据源当中都有，用的也是相同的英文名称
        all_spec_dict (dict): 分类变量的数据在用tree分类后返回的数字标签和相应的原始
            分类对应关系。 categorical_cut_bin()返回的all_spec_dict格式
        verbose (bool): default=True. If set True, will print the process logging
        ks_order: 'eventrate_order' 按照badrate排序后计算ks；'bin_order' 按照变量的分箱顺序计算ks

        Returns:
        woe_iv_df (pd.DataFrame): contains the following columns
        [u'数据源', u'指标英文', u'指标中文', u'数据类型', u'指标类型',\
        u'分箱', 'N', u'分布占比', 'WOE', u'逾期率', u'Bad分布占比',\
        u'Good分布占比', 'IV']
        """
        if all_spec_dict0 is None:
            spec_dict_flag = False
        else:
            spec_dict_flag = True
            all_spec_dict = deepcopy(all_spec_dict0)
            all_spec_dict = {k:v for k,v in list(all_spec_dict.items()) \
                                if var_dict.loc[var_dict['指标英文']==k, '数据类型'].iloc[0]=='varchar'}
        woe_iv_result = []
        if sample_weight is None:
            for col in X.columns.values:
                if verbose:
                    logging.log(18, col + ' woe calculation starts')
                woe_iv_result.append(self.calculate_woe(X[col], y,ks_order=ks_order))
        else:
            #X=pd.merge(X,sample_weight,left_index=True,right_index=True)
            for col in X.columns.values:
                woe_iv_result.append(self.calculate_woe_sw(X[col],y,sample_weight,col,ks_order=ks_order))
        woe_iv_df = pd.concat(woe_iv_result)
        woe_iv_df.loc[:, 'comment'] = 'useless_0.02_minus'
        woe_iv_df.loc[(woe_iv_df.IV <0.1) & (woe_iv_df.IV >= 0.02), 'comment'] = 'weak_predictor_0.02_0.1'
        woe_iv_df.loc[(woe_iv_df.IV <0.3) & (woe_iv_df.IV >= 0.1), 'comment'] = 'medium_predictor_0.1_0.3'
        woe_iv_df.loc[(woe_iv_df.IV <0.5) & (woe_iv_df.IV >= 0.3), 'comment'] = 'strong_predictor_0.3_0.5'
        woe_iv_df.loc[(woe_iv_df.IV >= 0.5), 'comment'] = 'strong_predictor_0.5_plus'
        woe_iv_df = woe_iv_df.rename(columns = {
                                        'var_code': '指标英文',
                                        'bin': '分箱',
                                        'PercentDist': '分布占比',
                                        'PercentBad': 'Bad分布占比',
                                        'PercentGood': 'Good分布占比',
                                        'EventRate': '逾期率',
                                        'NBad': '坏样本数量',
                                        'cumBad': 'Cumulative Bad Rate',
                                        'cumGood': 'Cumulative Good Rate',
                                        'cum_diff': 'Cumulative Rate Difference'
                                    })
        if var_dict is None:
            var_dict_flag = False
            woe_result = woe_iv_df
        else:
            var_dict_flag = True
            woe_result = var_dict.loc[:, ['数据源', '指标英文', '指标中文', '数据类型', '指标类型']]\
                        .merge(woe_iv_df, on='指标英文', how='right')

        if spec_dict_flag:
            # 加上分类变量的数据在用tree分类后返回的数字标签和相应的原始分类对应关系。
            woe_result.loc[:, '分箱对应原始分类'] = None
            for col, col_spec in list(all_spec_dict.items()):
                for new_label, original_x in list(col_spec.items()):
                    col_spec[str(new_label)] = ', '.join([str(i) for i in original_x])
                woe_result.loc[woe_result['指标英文'] == col, '分箱对应原始分类'] = woe_result.loc[woe_result['指标英文'] == col, '分箱'].astype('str')
                woe_result.loc[woe_result['指标英文'] == col, '分箱对应原始分类'] = woe_result.loc[woe_result['指标英文'] == col, '分箱对应原始分类'].replace(col_spec)
            if var_dict_flag:
                reorder_cols = ['数据源', '指标英文', '指标中文', '数据类型', '指标类型',\
                            '分箱', '分箱对应原始分类', 'N', '分布占比', '坏样本数量', '逾期率', 'WOE', \
                            'Bad分布占比', 'Good分布占比', 'Cumulative Bad Rate',\
                            'Cumulative Good Rate', 'Cumulative Rate Difference',\
                            'IV', 'KS', 'comment', 'sort_val']
            else:
                reorder_cols = ['指标英文','分箱', '分箱对应原始分类', 'N', '分布占比', '坏样本数量', '逾期率', 'WOE', \
                            'Bad分布占比', 'Good分布占比', 'Cumulative Bad Rate',\
                            'Cumulative Good Rate', 'Cumulative Rate Difference',\
                            'IV', 'KS', 'comment', 'sort_val']
        else:
            if var_dict_flag:
                reorder_cols = ['数据源', '指标英文', '指标中文', '数据类型', '指标类型',\
                            '分箱', 'N', '分布占比', '坏样本数量', '逾期率', 'WOE', \
                            'Bad分布占比', 'Good分布占比', 'Cumulative Bad Rate',\
                            'Cumulative Good Rate', 'Cumulative Rate Difference',\
                            'IV', 'KS', 'comment', 'sort_val']
            else:
                reorder_cols = ['指标英文','分箱', 'N', '分布占比', '坏样本数量', '逾期率', 'WOE', \
                            'Bad分布占比', 'Good分布占比', 'Cumulative Bad Rate',\
                            'Cumulative Good Rate', 'Cumulative Rate Difference',\
                            'IV', 'KS', 'comment', 'sort_val']
        result = woe_result[reorder_cols]
        return result

    def produce_cat_cut(self, data, all_with_y, var_dict, set_label, rebin_spec, selected):
        """
        Wrapper function for applying binning to new data with selected variables
        and selected set (training, or testing)

        Args:
        data (pd.DataFrame): X data
        all_with_y (pd.DataFrame): 带有Y label和sample label的 dataframe。
        var_dict (pd.DataFrame): standard variable dictionary
        set_label (str): one of the values in all_with_y.loc[:, 'sample'] for filtering the dataset
        rebin_spec (dict): the binning spec for variables
        selected (list): the list of selected variable names

        Returns:
        X_cat (pd.DataFrame): X data after binning for the selected "set_label" data
        y (pd.Series): the y label for the selected "set_label" data
        """
        # var_dict= pd.read_excel(os.path.join(DATA_PATH, '变量字典.xlsx'), encoding='utf-8', sheetname = '变量字典')
        all_data = data.merge(all_with_y[['apply_id', 'Y', 'sample']], on='apply_id')
        all_data = all_data.loc[all_data['sample']==set_label, :].drop('sample', 1)
        td_united = set(var_dict.loc[var_dict['数据源']=='同盾联合建模', '指标英文'])
        td_united = list(td_united.intersection(set(selected)))
        if len(set(rebin_spec.keys()).intersection(set(td_united))) == len(td_united):
            X = all_data[selected].copy()
            X_cat = self.convert_to_category(X, var_dict, rebin_spec)
        else:
            selected_non_td_united = list(set(selected) - set(td_united))
            X1 = all_data[selected_non_td_united].copy()
            X2 = all_data[td_united].copy()
            y = all_data.Y.copy()
            X_cat = self.convert_to_category(X1, var_dict, rebin_spec)
            X_cat = pd.concat([X_cat, X2], 1)
        y = all_data.Y
        return X_cat, y


    def create_rebin_spec(self, woe_iv_df, cat_spec_dict, original_cat_spec, missing_alone=True):
        """
        把分类变量的 level:grouping:label_number 或 level:label_number综合到一起以及
        连续变量的分箱

        Args:
        woe_iv_df (pd.DataFrame): woe_iv_df data frame, 标准输出
        cat_spec_dict (dict): categorical variable binning 的输出
        original_cat_spec (dict): categorical variable binning 的输出
        missing_alone: 是否将-8887等缺失值单独处理为一箱，默认单独处理，传入 obtain_boundaries
        # woe_iv = pd.read_excel(os.path.join(RESULT_PATH, 'WOE_bin/%s_woe_iv_df.xlsx' % model_label), encoding='utf-8')
        # cat_spec_dict = load_data_from_pickle(RESULT_PATH, 'spec_dict/%s_spec_dict.pkl' % model_label)
        # original_cat_spec = load_data_from_pickle(RESULT_PATH, 'spec_dict/%s_encoding_map.pkl' % model_label)

        Returns:
        cat_spec_dict (dict): 合并好的dictionary
        """
        cat_spec_dict_copy = deepcopy(cat_spec_dict)
        for col, col_spec in list(cat_spec_dict_copy.items()):
            for cat_name, cat_spec in list(col_spec.items()):
                if isinstance(cat_name, str) and isinstance(cat_spec, str):
                    del col_spec[cat_name]
        cols = woe_iv_df.loc[woe_iv_df['数据类型']!='varchar', '指标英文'].tolist()
        cols = set(cols) - set(cat_spec_dict_copy.keys())
        #处理数值型变量
        for col in cols:
            var_bin = woe_iv_df.loc[woe_iv_df['指标英文'] == col, '分箱'].copy()
            cat_spec_dict_copy[col] = self.obtain_boundaries(var_bin, missing_alone=missing_alone)
        original_cat_cols = list(set(original_cat_spec.keys()) - set(cat_spec_dict_copy.keys()))
        if original_cat_cols:
            for col in original_cat_cols:
                cat_spec_dict_copy[col] = {v:k for k, v in list(original_cat_spec[col].items())}

        return cat_spec_dict_copy


    def generate_bin_map(self, bin_boundaries):
        """
        For numerical variables, 分箱转换成对应的数值.

        Args:
        bin_boundaries (dict): 分箱spec
            {
                'other_categories': [-8888.0, 0.0],
                'cut_boundaries': [0.0, 550.0, 2110.0, 5191.0, 9800.0, 16000.0, 28049.0, 54700.0, np.inf]
            }

        Returns:
        bin_to_label (dict): key为分箱，value为对应的数值
            {'(0.0, 1.0]': 0, '(1.0, inf]': 1}
        """
        a = pd.Series(0)
        # 这样拿到的是最全的分箱值，不依赖数据里是否有这一分箱。且顺序是排好的
        complete_bins = [str(i) for i in pd.cut(a, bin_boundaries['cut_boundaries']).cat.categories]
        complete_bins_list = sorted(bin_boundaries['other_categories']) + list(complete_bins)
        complete_bins_df = pd.Series(complete_bins_list)
        reversed_map = complete_bins_df.to_dict()
        bin_to_label = {v:k for k,v in reversed_map.items()}
        return bin_to_label




    def xgboost_data_derive(self, X, y, var_dict,rebin_spec_adjusted=None,num_max_bins=10, verbose=True):
        """
        准备xgboost的数据，将原始数据自动分箱，分类变量做dummy variable

        Args:
        X (pd.DataFrame): 原始X数据， index为如applyid一样的unique identifier
        y (pd.Series): Y data, index为如applyid一样的unique identifier,与X对齐
        var_dict (pd.DataFrame): 标准化变量字典
        num_max_bins (int): default=10
        verbose (bool): default=True, 会输出运算过程信息。If False, 无过程信息输出
        rebin_spec_adjusted(dict):分箱对应的边界值
        X_cat(pd.DataFrame):X变量的分箱文件
        Return:
        X_derived(pd.DataFrame): 衍生后的数据包含原始变量值，自动分箱值，且分箱转换好数值，
            分类变量的variable
        auto_rebin_spec (dict): 自动分箱的分箱边界
        bin_to_label (dict): 自动分箱的分箱对应的数值转化, key为变量名，value为dict，
            dict的key为分箱，value为对应转换数值
        """
        if rebin_spec_adjusted is not None:

            auto_rebin_spec = rebin_spec_adjusted

            X_auto_binned = self.convert_to_category(X, var_dict, auto_rebin_spec,
                                        verbose=True,replace_value='min_value')

        else:
            X_auto_binned, all_encoding_map, all_spec_dict = self.binning(X, y,
                var_dict, num_max_bins=num_max_bins, verbose=False, missing_alone=False)

            woe_iv_df = self.calculate_woe_all(X_auto_binned, y, var_dict,
                                                all_spec_dict, verbose=False)
            auto_rebin_spec = self.create_rebin_spec(woe_iv_df, all_spec_dict,
                                                all_encoding_map, missing_alone=False)

        bin_to_label = {}
        for col, bin_boundaries in auto_rebin_spec.items():
            if col not in X_auto_binned.columns:
                continue

            if var_dict.loc[var_dict['指标英文']==col, '数据类型'].iloc[0] != 'varchar':
                bin_to_label[col] = self.generate_bin_map(bin_boundaries)
                try:
                    if verbose:
                        logging.log(18, "Numerical variable " + col + " starts to convert bin to label")

                    X_auto_binned[col] = X_auto_binned[col].astype(str)\
                                                           .replace(bin_to_label[col])\
                                                           .astype(int) #按说应该都是int，但是double check

                    if verbose:
                        logging.log(18, "Numerical variable " + col + " completes to convert bin to label")
                except:
                    logging.error("Numerical variable " + col + 'failed conversion!')

            else:
                # 如果为varchar变量，按说应该已经是数值了，只是类型为str，转为int
                try:
                    if verbose:
                        logging.log(18, "Categorical variable " + col + " starts to convert bin to label")

                    X_auto_binned[col] = X_auto_binned[col].astype(int)

                    if verbose:
                        logging.log(18, "Categorical variable " + col + " completes to convert bin to label")
                except:
                    logging.error("Categorical variable " + col + 'failed conversion!')

        # 有些categorical指标因为类别太小所以没有分箱弄到一起，所以就不会出现在auto_rebin_spec里面
        # 同时这些原始值的字段是不能放进xgboost模型里的，因为不是int或float
        for col in X_auto_binned.columns:
            if col not in auto_rebin_spec:
                try:
                    X_auto_binned[col] = X_auto_binned[col].astype(int)
                except:
                    bin_to_label[col] = dict(zip(X_auto_binned[col].unique(), range(X_auto_binned[col].nunique())))
                    X_auto_binned[col] = X_auto_binned[col].replace(bin_to_label[col])\
                                                           .astype(int)

        str_varlist = list(set(X.columns).intersection(var_dict.loc[var_dict['数据类型']=='varchar', '指标英文'].unique()))
        if len(str_varlist) > 0:
            X_dummy = pd.get_dummies(X[str_varlist], columns=str_varlist, drop_first=False, prefix_sep='DUMMY')
            # feature_names should not contain [, ] or <, replace with (, ) and lt
            # column names 不能含有特殊符号。
            # xgboost转dummy可能会有些值带contain [, ] or <不能作为变量名称
            new_indexs = []
            old_columns = list(X_dummy.columns)
            for col in range(len(old_columns)):
                new_col = old_columns[col].split('DUMMY')[0] + 'DUMMY'+str(col)
                new_indexs.append(new_col)

            dummy_var_name_map = dict(zip(new_indexs, X_dummy.columns))

            X_dummy.columns = new_indexs

            X_auto_binned.columns = [str(i) + '_binnum' for i in X_auto_binned.columns]

            X_derived = X.merge(X_auto_binned, left_index=True, right_index=True)\
                         .merge(X_dummy, left_index=True, right_index=True)

            X_derived = X_derived.drop(str_varlist, 1)
        else:
            dummy_var_name_map = {}
            X_auto_binned.columns = [str(i) + '_binnum' for i in X_auto_binned.columns]
            X_derived = X.merge(X_auto_binned, left_index=True, right_index=True)


        return X_derived, auto_rebin_spec, bin_to_label, dummy_var_name_map

    def convert_xgboost_rebins_to_dummy(self,X_derived,X_cat,dummy_var_name_map,is_apply=False):
        """
        将xgboost的rebin数据制作成dummy数据，并与之前进行过xgboost衍生的数据进行合并

        Args:
        X_derived(pd.DataFrame): 衍生后的数据包含原始变量值，自动分箱值，且分箱转换好数值，
            分类变量的variable
        X_cat (pd.DataFrame): 自动分箱的分箱对应的分箱边界
        dummy_var_name_map(dict): dummy转化字典
        Return:
        (pd.DataFrame): 原有的xgboost衍生数据+X_cat转换dummy以后的数据
        """
        if not is_apply:
            X_cat_dummy = pd.get_dummies(X_cat)
            bin_dummy_map = {}
            for j in X_cat.columns:
                unique_values = X_cat[j].unique()
                for i in range(len(unique_values)):
                    bin_dummy_map[j+'_'+str(unique_values[i])] = j+'_'+'bin_dummy%s'%str(i)
            X_cat_dummy.rename(columns=bin_dummy_map,inplace=True)
            reversed_map = {}
            for i in bin_dummy_map.keys():
                reversed_map[bin_dummy_map[i]] = i
            dummy_var_name_map.update(reversed_map)
        else:
            X_cat_dummy = pd.get_dummies(X_cat)
            reversed_map = {}
            for i in dummy_var_name_map.keys():
                reversed_map[dummy_var_name_map[i]] = i
            X_cat_dummy.rename(columns=reversed_map,inplace=True)
            X_derived.rename(columns=reversed_map,inplace=True)
        return pd.merge(X_derived,X_cat_dummy,left_index=True,right_index=True)

    def apply_xgboost_data_derive(self, X, var_dict, auto_rebin_spec, bin_to_label,
                                  verbose=True, dummy_var_name_map=None):
        """
        准备xgboost的数据，将原始数据自动分箱分箱转换成数值，分类变量做dummy variable
        原始数据取值列名为原始名，自动分箱转换成数值列名为原始变量名+'_binnum'
        dummy变量的变量名为原始变量名+具体取值

        Args:
        X (pd.DataFrame): 原始X数据， index为如applyid一样的unique identifier
        var_dict (pd.DataFrame): 标准化变量字典
        auto_rebin_spec (dict): 自动分箱的分箱边界
        bin_to_label (dict): 自动分箱的分箱对应的数值转化, key为变量名，value为dict，
            dict的key为分箱，value为对应转换数值
        verbose (bool): default=True, 会输出运算过程信息。If False, 无过程信息输出
        dummy_var_name_map (dict): dummy转化字典

        Return:
        X_derived(pd.DataFrame): 衍生后的数据包含原始变量值，自动分箱值，且分箱转换好数值，
            分类变量的variable
        """
        X_auto_binned = self.convert_to_category(X, var_dict, auto_rebin_spec,
                                        verbose=True,replace_value='min_value')

        for col, bin_boundaries in auto_rebin_spec.items():
            if col not in X_auto_binned.columns:
                continue

            if var_dict.loc[var_dict['指标英文']==col, '数据类型'].iloc[0] != 'varchar':
                if col not in bin_to_label:
                    continue

                try:
                    if verbose:
                        logging.log(18, "Numerical variable " + col + " starts to convert bin to label")

                    X_auto_binned[col] = X_auto_binned[col].astype(str)\
                                                           .replace(bin_to_label[col])\
                                                           .astype(int) #按说应该都是int，但是double check

                    if verbose:
                        logging.log(18, "Numerical variable " + col + " completes to convert bin to label")
                except:
                    logging.error("Numerical variable " + col + 'failed conversion!')

            else:
                # 如果为varchar变量，按说应该已经是数值了，只是类型为str，转为int
                try:
                    if verbose:
                        logging.log(18, "Categorical variable " + col + " starts to convert bin to label")

                    X_auto_binned[col] = X_auto_binned[col].astype(int)

                    if verbose:
                        logging.log(18, "Categorical variable " + col + " completes to convert bin to label")
                except:
                    logging.error("Categorical variable " + col + 'failed conversion!')


        # 有些categorical指标因为类别太小所以没有分箱弄到一起，所以就不会出现在auto_rebin_spec里面
        # 同时这些原始值的字段是不能放进xgboost模型里的，因为不是int或float
        for col in X_auto_binned.columns:
            if col not in auto_rebin_spec:
                try:
                    X_auto_binned[col] = X_auto_binned[col].astype(int)
                except:
                    X_auto_binned[col] = X_auto_binned[col].replace(bin_to_label[col])\
                                                           .astype(int)

        str_varlist = list(set(X.columns).intersection(var_dict.loc[var_dict['数据类型']=='varchar', '指标英文'].unique()))
        if len(str_varlist) > 0:
            X_dummy = pd.get_dummies(X[str_varlist], columns=str_varlist, drop_first=False, prefix_sep='DUMMY')
            # feature_names should not contain [, ] or <, replace with (, ) and lt
            # column names 不能含有特殊符号。
            # xgboost转dummy可能会有些值带contain [, ] or <不能作为变量名称
            '''

            new_indexs = []
            for col in X_dummy.columns:
                new_col = col.replace('[','(')\
                             .replace(']',')')\
                             .replace('<','lt')\
                             .replace(',', '')\
                             .replace('>', 'mt')

                new_indexs.append(new_col)

            X_dummy.columns = new_indexs
            '''
            # 取代上边的方法直接将dummy_var_name_map应用在test数据上
            if dummy_var_name_map:
                reversed_map = {j:i for i,j in dummy_var_name_map.items()}
                X_dummy.rename(columns=reversed_map,inplace=True)

            X_auto_binned.columns = [str(i) + '_binnum' for i in X_auto_binned.columns]

            X_derived = X.merge(X_auto_binned, left_index=True, right_index=True)\
                         .merge(X_dummy, left_index=True, right_index=True)

            X_derived = X_derived.drop(str_varlist, 1)
        else:
            X_auto_binned.columns = [str(i) + '_binnum' for i in X_auto_binned.columns]
            X_derived = X.merge(X_auto_binned, left_index=True, right_index=True)
        return X_derived


    def xgboost_obtain_raw_variable(self, xgboost_derived_variable, var_dict):
        """
        xgboost derived 的变量名称追溯回原始的变量名。比如一个变量做了dummy转化后的
        某一个分类（e.g. province_北京市)需要找出这个字段的原始变量是province
        """
        if xgboost_derived_variable in var_dict['指标英文'].tolist():
            return ('num_vars_origin', xgboost_derived_variable)


        if '_binnum' in xgboost_derived_variable:
            result = xgboost_derived_variable.replace('_binnum', '')
            if result in var_dict['指标英文'].tolist():
                return ('bin_vars', result)


        name_splits = xgboost_derived_variable.split('DUMMY')
        result = name_splits[0]
        if result in var_dict['指标英文'].tolist():
            return ('dummy_vars', result)

        return (None, None)


# add by jzw
    def add_sample_weight(self,data,segment_label,sample_weight):
        """
        Args:
        data(DataFrame):包含x和y变量的dataframe，其index为applyid且为unique identifier；
        sample_weight(dict):不同segment区分好坏样本抽样的sample_weight,如下所示：
        sample_weight={'Bad Channel':{'good_sample':10,'bad_sample':5},\
               'Good Channel':{'good_sample':8,'bad_sample':3},\
              'Normal Channel':{'good_sample':9,'bad_sample':4}}

        Retruns:
        data_fianl(DataFrame):包含sample_weight的建模样本
        """
        if len(sample_weight)==0:
            print('未进行抽样取样')
            data['sample_weight']=1
            data_fianl=data
        else:
            data_good=data[data['y']==0]
            data_bad=data[data['y']==1]
            sample_data=[]
            for seg,sw in sample_weight.items():
                g_dt=data_good[data_good[segment_label]==seg]
                g_dt['sample_weight']=sw['good_sample']
                sample_data.append(g_dt)
            for seg,sw in sample_weight.items():
                b_dt=data_bad[data_bad[segment_label]==seg]
                b_dt['sample_weight']=sw['bad_sample']
                sample_data.append(b_dt)
            #print(len(sample_data))
            data_fianl=pd.concat(sample_data)
        return data_fianl







class Performance(object):
    def __init__(self):
        pass

    def calculate_auc(self, y, prob):
        """
        计算AUC

        Args:
        y (pd.Series): y标签
        prob (pd.Series): 预测概率

        Returns:
        roc_auc_train (float): auc值
        """
        fpr_train, tpr_train, thresholds = metrics.roc_curve(y, prob)
        roc_auc_train = metrics.auc(fpr_train, tpr_train)
        return roc_auc_train

    def calculate_ks_by_score(self, y, score):
        """
        将分数切分成20等分计算ks

        Args:
        y (pd.Series): y标签
        prob (pd.Series): 预测概率

        Returns:
        roc_auc_train (float): auc值
        """
        decile = self.calculate_ks_by_decile(score, np.array(y), 'decile', 20)
        return decile.KS.max()


    def calculate_score(self, coeficients, data, job, var_dict=None, \
                        manual_adjust_var_bin=None, point_double_odds=75, \
                        base_odds=15, base_points=660):
        """
        根据job的不同计算总分，或是评分卡。

        Args：
        coefficients (pd.DataFrame): 必须包括'var_code' (即指标英文)，'beta'

        data (pd.DataFrame):
        If job == 'overall_score', data需要是train或是test set，且值为woe的dataframe，
        data:
           c_mateName_provided     c_sex     n_age
        0            -0.032069  0.082254 -0.020282
        1             0.045955  0.082254 -0.020282
        2             0.045955  0.082254 -0.020282
        3             0.045955  0.082254 -0.020282
        4             0.045955 -0.222418 -0.020282

        If job in ['var_score', 'overall_score_adjust'], data需要是woe带有bin命名的
        data frame。格式为long table。woe_iv_df的输出格式，不用重新命名。
        data:
            指标英文	分箱	           N	分布占比	     WOE	   逾期率 	  Bad分布占比
        0	CSSS001	-8888.0	         655	 0.031365	-0.122253	0.021374	0.027833
        1	CSSS001	(-inf, 590.0]	11837	0.566825	0.092448	0.026358	0.620278
        2	CSSS001	(590.0, 912.0]	4218	0.201982	0.004055	0.024182	0.202783
        3	CSSS001	(912.0, inf]	4173	0.199828	-0.299047	0.017973	0.149105
        4	CSSP003	-8888.0	         655	0.031365	-0.122253	0.021374	0.027833

        job (str): ['overall_score', 'var_code', 'overall_score_adjust']。
            job == 'overall_score'时，会计算出overall model score，前提是没有变量的bin需要重新打分。
            job == 'var_score'时，计算每个变量区间取分值，即评分卡
            job == 'overall_score_adjust'时，计算每个变量区间取分，并按照manual_adjust_var_bin
            里列出的variable bin调整分数

        var_dict (pd.DataFrame): 标准变量字典. 请做一些初期的数据源筛选。因为有些变量在
            不同数据源当中都有，用的也是相同的英文名称

        manual_adjust_var_bin(list): 需要手动调整分数为0分的var_code的bin
        [('n_networkTime6', '-8888'), ('c_province', 'loc_other')]

        point_double_odds (int): default = 75
        base_odds (int): 因为我们是反着的，逾期概率越高的，分数应该越低，因此base_odds = p(good) / p(bad)
            default = 15
        base_points (int): default = 660

        Returns:
        total_score (pd.Series): job == 'overall_score_adjust'时，返回计算出的总分
        result (pd.DataFrame): job in ['var_score', 'overall_score_adjust'], 返回评分卡.
            评分卡的基数是intercept相对应的分，然后在着基础上每个变量加减分。这样能确保WOE是零的时候，加减分也是0分。
            包括：[u'数据源', u'指标英文', u'指标中文', u'数据类型', u'指标类型',\
                u'分箱', 'N', u'分布占比', 'WOE', u'逾期率', u'Bad分布占比', u'变量打分']
        """
        n = len(coeficients) - 1
        intercept = coeficients.loc[coeficients.var_code.str.contains('Intercept|const'), 'beta'].iloc[0]
        factor = point_double_odds / np.log(2)
        offset = base_points - point_double_odds * (np.log(base_odds) / np.log(2))
        var_names = coeficients.loc[~coeficients.var_code.str.contains('Intercept|const'), 'var_code'].tolist()
        if job == 'overall_score':
            beta_vector = coeficients.loc[~coeficients.var_code.str.contains('Intercept|const'), 'beta']
            woe_matrix = data.loc[:, var_names]
            log_odds = intercept + pd.Series(np.dot(woe_matrix, beta_vector))
            # flipped the odss to: P(good)/P(bad),其实就是加个负号
            log_odds_good2bad = np.log(1 / np.exp(log_odds))
            total_score = factor * log_odds_good2bad + offset
            return total_score

        if job in ['var_score', 'overall_score_adjust']:
            data = data.rename(columns={'var_code': '指标英文', 'bin': '分箱'})
            intercept_score = base_points + factor * (- intercept - np.log(base_odds))
            beta_df = coeficients.loc[~coeficients.var_code.str.contains('Intercept|const'), ['var_code','beta']]
            data2 = data.loc[data['指标英文'].isin(var_names), :]
            data2 = data2.merge(beta_df, left_on='指标英文', right_on='var_code').drop('var_code', 1)
            # data2.loc[:, 'partial_log_odds'] = intercept * 1.0 / n + data2.beta * data2.WOE
            # data2.loc[:, 'partial_log_odds_good2bad'] =  - data2.partial_log_odds
            data2.loc[:, 'var_score'] = - factor * (data2.beta * data2.WOE)
            intercept_score_df = pd.Series({'var_code': 'intercept', 'var_score': intercept_score})
            intercept_score_df = intercept_score_df.to_frame()\
                                                   .transpose()\
                                                   .rename(columns={'var_code': '指标英文'})
            # 减去basepoint的部分是为了让最后每个变量的分是基于660加或者减的到总分，而不是每个变量的评分总和得到总分
            # data2.loc[:, 'var_score'] = factor * data2.partial_log_odds_good2bad + offset * 1.0 / n - base_points * 1.0 / n

            if job == 'overall_score_adjust':
                for var_adjust in manual_adjust_var_bin:
                    data2.loc[(data2.var_code==var_adjust[0]) & (data2.bin==var_adjust[1]), 'var_score'] = 0

            # return data2[['var_code', 'bin', 'N', 'PercentDist', 'WOE', 'EventRate', 'PercentBad','var_score']]
            result = pd.concat([intercept_score_df,\
                            data2[['指标英文', '分箱', 'N', '分布占比', 'WOE', '逾期率', 'Bad分布占比', 'var_score']]\
                            ], ignore_index=True)\
                        .rename(columns={'var_score': '变量打分'})
            result = var_dict[['数据源', '指标英文', '指标中文', '数据类型', '指标类型']]\
                            .merge(result, on='指标英文', how='right')
            result.loc[:, '变量打分'] = result.loc[:, '变量打分'].astype('float').round()
            reorder_cols = ['数据源', '指标英文', '指标中文', '数据类型', '指标类型',\
                '分箱', 'N', '分布占比', 'WOE', '逾期率', 'Bad分布占比', '变量打分']
            return result[reorder_cols]


    def calculate_score_by_scrd(self, bin_data, var_score_data, verbose=True):
        """
        根据已经算好的变量加减分评分卡在已经转成bin的X上面计算total score，对于需要调整评分
        卡上个别变量区间打分情况的可以用这个来计算最终的model score。

        Args:
        bin_data (pd.DataFrame): 变量宽表，值是bin值，而不是woe值
        var_score_data (pd.DataFrame):变量打分表,需要包含列为['var_code', 'bin', 'var_score']
        E.g.
        var_score_data = calculate_score(coeficients, woe_iv_df, job='overall_score_adjust', \
                                        manual_adjust_var_bin=[('n_networkTime6', '-8888'), ('c_province', 'loc_other'), ('n_longTimeShutdown', '-8888')])
        verbose (bool): default=True. If set True, will print the process logging

        Returns:
        score (pd.Series): 计算好的分数
        """
        var_score_data = var_score_data.rename(columns={'指标英文':'var_code', '分箱':'bin', '变量打分':'var_score'})
        bin_data = bin_data.copy()
        var_names = [i for i in var_score_data.var_code.unique() if 'intercept' not in i]
        for var_code in var_names:
            if verbose:
                logging.log(18, var_code + ' scoring starts')

            replace_map = var_score_data.loc[var_score_data.var_code==var_code, ['bin', 'var_score']].copy()
            replace_map.index = replace_map.bin
            replace_map.var_score = replace_map.var_score.astype(int)
            try:
                bin_data.loc[:, var_code] = bin_data.loc[:, var_code].astype('str').replace(replace_map.var_score.to_dict()).astype(float).replace({-9999:0})
            except:
                bin_data.loc[:, var_code] = bin_data.loc[:, var_code].astype('str').replace(replace_map.var_score.to_dict()).replace({-9999:0})
                value_type = bin_data.loc[:, var_code].apply(lambda x: str(type(x)))
                not_converted = value_type.str.contains('str')
                if sum(not_converted) > 0:
                    print('=========')
                    print('WARNING: new category shows, and set var score to 0')
                    print(bin_data.loc[:, var_code].loc[not_converted].value_counts())
                    bin_data.loc[value_type.str.contains('str'), var_code] = 0
        # score = bin_data[var_names].sum(1) + 660
        selected_score_data = bin_data[var_names].astype(float)
        score = selected_score_data.sum(1) + var_score_data.loc[var_score_data.var_code.str.contains('intercept'),\
                                                                'var_score'].iloc[0]
        return selected_score_data, np.round(score)

    def p_to_score(self, p, PDO=75.0, Base=660, Ratio=1.0/15.0):
        """
        逾期概率转换分数

        Args:
        p (float): 逾期概率
        PDO (float): points double odds. default = 75
        Base (int): base points. default = 660
        Ratio (float): bad:good ratio. default = 1.0/15.0

        Returns:
        化整后的模型分数
        """
        B = 1.0*PDO/log(2)
        A = Base+B*log(Ratio)
        score=A-B*log(p/(1-p))
        return round(score,0)

    def score_to_p(self, score, PDO=75.0, Base=660, Ratio=1.0/15.0):
        """
        分数转换逾期概率

        Args:
        score (float): 模型分数
        PDO (float): points double odds. default = 75
        Base (int): base points. default = 660
        Ratio (float): bad:good ratio. default = 1.0/15.0

        Returns:
        转化后的逾期概率
        """
        B = 1.0*PDO/log(2)
        A = Base+B*log(Ratio)
        alpha = (A - score) / B
        p = np.exp(alpha) / (1+np.exp(alpha))
        return p


    def calculate_ks_by_decile(self, score, y, job, q=10, score_bin_size = 25, manual_cut_bounds=[], score_type='raw'):
        """
        可同时计算decile analysis和Runbook analysis

        Args:
        score (pd.Series): 计算好的模型分
        y (pd.Series): 逾期事件label
        job (str): ['decile', 'runbook'], decile时，将会把score平分成q份， runkbook时，
        将会把平分分成25分一档的区间。有一种runbook是decile时，q=20
        q (int): default = 10, 将score分成等分的数量
        manual_cut_bounds (list): default = [], 当需要手动切分箱的时候，可以将分箱的bounds
            传入。
        score_type (str): ['raw', 'binned']. default='raw'. if 'raw', 传入的 score 是原始分数, if 'binned',传入的是分箱好的数据

        Returns:
        r (pd.DataFrame): 按照score分箱计算的EventRate, CumBadPct, CumGoodPct等，用来
        放置于model evaluation结果里面。
        """
        if score_type == 'raw':
            score = np.round(score)
            if job == 'decile':
                if len(manual_cut_bounds) == 0:
                    decile_score = pd.qcut(score, q=q, duplicates='drop', precision=0).astype(str) #, labels=range(1,11))
                else:
                    decile_score = pd.cut(score, manual_cut_bounds, precision=0).astype(str)

            if job == 'runbook':
                if len(manual_cut_bounds) == 0:
                    min_score = int(np.round(min(score)))
                    max_score = int(np.round(max(score)))
                    score_bin_bounardies = list(range(min_score, max_score, score_bin_size))
                    score_bin_bounardies[0] = min_score - 0.001
                    score_bin_bounardies[-1] = max_score
                    decile_score = pd.cut(score, score_bin_bounardies, precision=0).astype(str)
                else:
                    decile_score = pd.cut(score, manual_cut_bounds, precision=0).astype(str)
        else:
            decile_score = score.astype(str)

        r = pd.crosstab(decile_score, y).rename(columns={0: 'N_nonEvent', 1: 'N_Event'})
        if 'N_Event' not in r.columns:
            r.loc[:, 'N_Event'] = 0
        if 'N_nonEvent' not in r.columns:
            r.loc[:, 'N_nonEvent'] = 0
        r.index.name = None
        r['rank'] = r.index
        r['rank'] = r['rank'].apply(lambda x:float(x[1:].split(',')[0]))
        r.sort_values(by='rank',inplace=True)
        r.drop(['rank'],1,inplace=True)
        r.loc[:, 'N_sample'] = decile_score.value_counts()
        r.loc[:, 'EventRate'] = r.N_Event * 1.0 / r.N_sample
        r.loc[:, 'Distribution'] = r.N_sample * 1.0 / r.N_sample.sum()
        r.loc[:, 'BadPct'] = r.N_Event * 1.0 / sum(r.N_Event)
        r.loc[:, 'GoodPct'] = r.N_nonEvent * 1.0 / sum(r.N_nonEvent)
        r.loc[:, 'CumBadPct'] = r.BadPct.cumsum()
        r.loc[:, 'CumGoodPct'] = r.GoodPct.cumsum()
        r.loc[:, 'KS'] = np.round(r.CumBadPct - r.CumGoodPct, 4)
        r.loc[:, 'odds(good:bad)'] = np.round(r.N_nonEvent*1.0 / r.N_Event, 1)
        r = r.reset_index().rename(columns={'index': '分箱',
                                            'N_sample': '样本数',
                                            'N_nonEvent': '好样本数',
                                            'N_Event': '坏样本数',
                                            'EventRate': '逾期率',
                                            'Distribution':'分布占比',
                                            'BadPct': 'Bad分布占比',
                                            'GoodPct': 'Good分布占比',
                                            'CumBadPct': '累积Bad占比',
                                            'CumGoodPct': '累积Good占比'
                                            })
        reorder_cols = ['分箱', '样本数', '好样本数', '坏样本数', '逾期率','分布占比',\
                        'Bad分布占比', 'Good分布占比', '累积Bad占比', '累积Good占比',\
                        'KS', 'odds(good:bad)']
        result = r[reorder_cols]
        result.loc[:, '分箱'] = result.loc[:, '分箱'].astype(str)
        return result

    def calculate_score_by_xgb_model_result(self, data, xgb_model_result):
        """
        通过xgb模型结果计算data的xgb对应的prob以及score

        Args:
        data (pd.DataFrame): 带有变量的数据
        xgb_model_result (dict): xgboost的模型结果

        Returns:
        data_XGB:data进行xgboost打分后的概率以及打分
        """
        # 拿到xgb模型中的features
        XGB_features = xgb_model_result['model_final'].feature_names
        # 将数据转换为xgboost专用格式
        data_XGB_matrix = xgb.DMatrix(data[XGB_features])
        # 得出概率
        data_XGB_p = pd.Series(xgb_model_result['model_final']\
                                      .predict(data_XGB_matrix)\
                                      ,index = data.index)
        data_XGB = data_XGB_p.to_frame('prob')
        # 将概率转化为打分
        data_XGB['xgbScore'] = data_XGB['prob'].apply(self.p_to_score)
        return data_XGB

    def score_dist(self, data, score_bins):
        """
        计算score分布。将score按照score_bins分箱，然后每个放款月（ddmonth）每个分数段的
        申请数

        Args:
        data (pd.DataFrame): 必须含有score, ddmonth列
        score_bins (list): 将score分箱的取值区间范围

        Returns:
        score_dist (pd.DataFrame): 每个放款月（ddmonth）每个分数段的申请数

        Example：
        min_score = score.min()
        max_score = score.max()
        [-np.inf, 550, 600, 650, 700, 750, np.inf]
        """
        min_score = score.min()
        max_score = score.max()
        data.loc[:, 'bin']  = pd.cut(data.score, score_bins)
        data.loc[:, 'bin'] = data.bin.astype('str')
        score_dist = data.groupby(['ddmonth', 'bin']).apply_id.nunique().reset_index()
        return score_dist


    def psi(self, before, after, raw_cat=True):
        """
        计算PSI

        Args:
        before (pd.Series): 基准时点分箱好的数据
        after (pd.Series): 比较时点的分箱好的数据
        raw_cat (bool): default=True. 传入的数据为分箱好的数据。if False, 传入的数据是
            value_counts 好的，比如decile表格的现成的

        Returns:
        combined (pd.DataFrame): 对齐好的before and after 占比数据
        psi_value (float): 计算的PSI值
        """
        #object在.sort_index()会报错，修改为str
        if before.dtype=='object':
            before=before.astype(str)
        if after.dtype=='object':
            after=after.astype(str)

        if raw_cat:
            before_ct = before.value_counts().sort_index()
            after_ct = after.value_counts().sort_index()
        else:
            before_ct = before
            after_ct = after

        before_pct = before_ct * 1.0 / len(before)
        after_pct = after_ct * 1.0 / len(after)

        if type(before_pct) == pd.Series:
            before_pct = before_pct.to_frame('before_pct')
        if type(after_pct) == pd.Series:
            after_pct = after_pct.to_frame('after_pct')

        if type(before_ct) == pd.Series:
            before_ct = before_ct.to_frame('before_ct')
        if type(after_ct) == pd.Series:
            after_ct = after_ct.to_frame('after_ct')

        df_list = [before_ct, after_ct, before_pct, after_pct]
        # outer join是因为可能出现before， after数据各自分出的箱不全
        combined = reduce(lambda b, a: b.merge(a, left_index=True, right_index=True, how='outer'), \
                          df_list)
        # 保留两位小数，以防止在比较小占比的箱上，一点点的波动导致PSI大
        combined.loc[:, 'before_pct'] = combined.before_pct.round(2)
        combined.loc[:, 'after_pct'] = combined.after_pct.round(2)

        #处理某些分箱无数据造成psi inf
        psi_index_value = (np.log(combined.after_pct/combined.before_pct) * (combined.after_pct - combined.before_pct))
        psi_index_value = psi_index_value.replace(np.inf, 0)
        combined.loc[:, 'PSI'] = psi_index_value.sum()
        combined = combined.reset_index().rename(columns={'index': '分箱'})
        # 添加排序列并排序
        combined = BinWoe().order_bin_output(combined, '分箱')
        combined = combined.drop('sort_val', 1)

        return combined, psi_index_value.sum()


    def variable_psi(self, X_cat_train, X_cat_test, var_dict):
        """
        批量计算变量的PSI

        Args:
        X_cat_train (pd.DataFrame): 分箱好的train数据，或者基准时点的数据
        X_cat_test (pd.DataFrame): 分箱好的test数据，或者比较时点的数据

        Returns
        result (pd.DataFrame): 包含指标英文，PSI两列。
        """
        before_columns = X_cat_train.columns
        after_columns = X_cat_test.columns
        common_columns = set(before_columns).intersection(set(after_columns))
        psi_result = []
        for col in common_columns:
            logging.log(18, 'variable PSI for %s' % col)
            psi_df, _ = self.psi(X_cat_train[col], X_cat_test[col])
            psi_df.loc[:, '指标英文'] = col
            psi_result.append(psi_df)

        result = pd.concat(psi_result)
        var_dict = var_dict.loc[:, ['数据源', '指标英文', '指标中文', '数据类型', '指标类型']]

        result = result.merge(var_dict, on='指标英文')

        result = result.rename(columns={'before_ct': '基准时点_N', \
                                       'before_pct': '基准时点占比',\
                                       'after_ct': '比较时点_N',\
                                       'after_pct': '比较时点占比'})
        ordered_cols = ['数据源', '指标英文', '指标中文', '数据类型', '指标类型',\
                        '分箱', '基准时点_N', '基准时点占比', '比较时点_N',\
                        '比较时点占比', 'PSI']
        return result[ordered_cols]

    def score_psi(self, before_score, after_score, score_cut_bins):
        """
        分数的PSI

        Args:
        before_score (pd.Series): train score，或者基准时点的score
        after_score (pd.Series): test score, 或者比较时点的score
        score_cut_bins (list): the cutting boundaries for score binning

        Returns:
        result (float): 计算好的分数的PSI
        """

        before_score_cut = pd.cut(before_score, score_cut_bins)
        after_score_cut = pd.cut(after_score, score_cut_bins)

        psi_df, _ = self.psi(before_score_cut, after_score_cut)
        psi_df.loc[:, '指标英文'] = 'modelScore'
        psi_df = psi_df.rename(columns={'before_ct': '基准时点_N', \
                               'before_pct': '基准时点占比',\
                               'after_ct': '比较时点_N',\
                               'after_pct': '比较时点占比'})
        ordered_cols = ['指标英文', '分箱', '基准时点_N', '基准时点占比', '比较时点_N',\
                        '比较时点占比', 'PSI']

        return psi_df[ordered_cols]

    def score_ranking_psi_train_vs_oot(self,ks_decile_train,ks_decile_OOT):
        """
        通过ks_decile计算训练集和OOT的psi以及ranking变化对比

        Args:
            ks_decile_train (pd.DataFrame): 训练集decile
            ks_decile_OOT (pd.DataFrame): 测试集decile
        result
            score_ranking_train_vs_oot(pd.DataFrame)：训练集和OOT的psi以及ranking变化对比
        """
        ks_decile_train_part = ks_decile_train[['分箱','样本数','逾期率']]
        ks_decile_train_part['分布占比'] = ks_decile_train['样本数']/ks_decile_train['样本数'].sum()

        ks_decile_OOT['分布占比'] = ks_decile_OOT['样本数']/ks_decile_OOT['样本数'].sum()
        score_ranking_train_vs_oot = pd.concat([ks_decile_train_part[['分箱','样本数','逾期率','分布占比']]\
          ,ks_decile_OOT[['样本数','逾期率','分布占比']]],axis=1)

        # 生成一些字段,为了和变量的dataFrame匹配
        score_ranking_train_vs_oot.columns = \
        ['分箱', 'train_set样本量', 'train_set逾期率', 'train_set分布占比'\
         , 'OOT_set样本量', 'OOT_set逾期率', 'OOT_set分布占比']
        score_ranking_train_vs_oot['数据源'] = '模型分'
        score_ranking_train_vs_oot['指标英文'] = 'modelScore'
        score_ranking_train_vs_oot['指标中文'] = '模型分'
        score_ranking_train_vs_oot['指标类型'] = '/'
        score_ranking_train_vs_oot['分箱对应原始分类'] = None

        score_ranking_train_vs_oot['PSI'] = score_ranking_train_vs_oot.apply(lambda x : (x['OOT_set分布占比'] - x['train_set分布占比'])\
                           *log(x['OOT_set分布占比']/x['train_set分布占比'])\
                           if x['OOT_set分布占比']!=0 and x['train_set分布占比']!=0 else 0\
                           ,axis=1)
        return score_ranking_train_vs_oot

    def all_score_psi_summary(self,before_score,after_score,binNum,y,beforeName,afterName):
        """
        为训练集和测试集的good和bad分别进行PSI测算

        Args:
            before_score (pd.Series): train score，或者基准时点的score
            after_score (pd.Series): test score, 或者比较时点的score
            binNum (int): 将分数等分的箱数
            y(pd.Series)：before_score+after_score对应的y
            beforeName：基准时点名称
            Returns:比较时点名称
        result
            all_score_to_plot(pd.DataFrame)：所有数据的分数汇总以及对应标签
            score_psi_result_good(pd.DataFrame)：好人的PSI分数统计
            score_psi_result_bad(pd.DataFrame)：坏人的PSI分数统计
            bins(list):分箱边界
        """
        before_score_ = before_score.to_frame('score')
        before_score_.loc[:, 'group_label'] = beforeName
        after_score_ = after_score.to_frame('score')
        after_score_.loc[:, 'group_label'] = afterName
        all_score_to_plot = pd.concat([before_score_, after_score_])
        all_score_to_plot=all_score_to_plot.merge(y.to_frame('Y'),left_index=True,right_index=True)#部分y的index不在all_score_to_plot中
        all_score_to_plot.loc[all_score_to_plot.Y==1, 'label2'] = np.where(all_score_to_plot.loc[all_score_to_plot.Y==1, 'group_label']\
                                                                  ==beforeName, beforeName+'_bad', afterName+'_bad')
        all_score_to_plot.loc[all_score_to_plot.Y==0, 'label2'] = np.where(all_score_to_plot.loc[all_score_to_plot.Y==0, 'group_label']\
                                                              ==beforeName, beforeName+'_good', afterName+'_good')
        before_good_score = all_score_to_plot[(all_score_to_plot.group_label==beforeName)\
                                         &(all_score_to_plot.label2==beforeName+'_good')].score
        before_bad_score = all_score_to_plot[(all_score_to_plot.group_label==beforeName)\
                                         &(all_score_to_plot.label2==beforeName+'_bad')].score
        after_good_score = all_score_to_plot[(all_score_to_plot.group_label==afterName)\
                                         &(all_score_to_plot.label2==afterName+'_good')].score
        after_bad_score = all_score_to_plot[(all_score_to_plot.group_label==afterName)\
                                         &(all_score_to_plot.label2==afterName+'_bad')].score
        #划分数据区间
        bins = list(np.linspace(min(before_score.min(),after_score.min())\
                                ,max(before_score.max(),after_score.max()),(binNum+1)))

        score_psi_result_good = self.score_psi(before_good_score, after_good_score, score_cut_bins=bins)
        score_psi_result_bad = self.score_psi(before_bad_score, after_bad_score, score_cut_bins=bins)
        return all_score_to_plot,score_psi_result_good,score_psi_result_bad,bins
