# encoding=utf8
"""
Plotting functions

Owner： 胡湘雨

最近更新：2017-12-07
"""

import os
import simplejson as json
from itertools import cycle
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import metrics
from sklearn.linear_model import lasso_path, enet_path
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk


import utils3.metrics as mt
import utils3.misc_utils as mu
from utils3.data_io_utils import *
sns.set_style("whitegrid")



def plot_total_distribution(y, result_path, save=False):
    
    pct = pd.crosstab(y, 'sum').reset_index()
    
    ax = sns.countplot(x = y, label='count')
    ax_v2 = ax.twinx()
    ax_v2.plot(pct[y_col_name], round(pct['sum']/pct['sum'].sum(),2), 'r-', lw=2)
    ax_v2.set_yticks([])
    ax_v2.set_ylim(-0.1, 1.5)
    for idx, row in pct.iterrows():
#        print(row[y_col_name], row['sum'])
        ax.text(row[y_col_name], row['sum'], round(row['sum'], 2), color='black', ha='center',size=14)
        ax_v2.text(row[y_col_name], row['sum']/pct['sum'].sum(), round(row['sum']/pct['sum'].sum(), 2), color='red', va='baseline', size=14)
    plt.title('Good vs Bad')
    if save:
        plt.savefig(os.path.join(result_path, '1_Good_VS_BAD.png'))
    plt.show()

"""
Examples:

# saving multiple plots in one pdf file
pp = PdfPages(os.path.join(DATA_PATH, 'test.pdf'))
edap = eda_plot()
for y in ['v', 'j']:
    edap.violin_plot(x='ddmonth', y=y, hue='Y', data=data, pdf_obj=pp)

pp.close()
"""


class eda_plot(object):
    def __init__(self):
        pass
        # sns.set(style="whitegrid", palette="pastel", color_codes=True)

    def violin_plot(self, x, y, hue, data, pdf_obj):
        """
        For detail usage of args, see seaborn.violin_plot()

        Example:
        pdf_obj = PdfPages('test.pdf')
        eda_plot().violin_plot(x='ddmonth', y='cont_variable', hue='Y', data=data, pdf_obj=pdf_obj)
        """
        p = sns.violinplot(x=x, y=y, hue=hue, data=data, split=True,
                       inner="quart", palette={0: "g", 1: "r"})
        plt.title('EDA Violin Plot for %s' % y)
        plt.legend(title= hue, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)
        pdf_obj.savefig()
        plt.close()

    def count_pct_plot(self, x, hue, data, pdf_obj):
        """
        For detail usage of args, see seaborn.countplot()

        Example:
        pdf_obj = PdfPages('test.pdf')
        eda_plot().count_pct_plot(x='ddmonth', hue='cat_variable', data=data, pdf_obj=pdf_obj)
        """
        plot = sns.countplot(x=x, hue=hue, data=data)
        plt.title('EDA Count Plot for %s' % hue)
        plt.legend(title= hue, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)
        plt.ylabel('Number of occurrences')
        ### Adding percents over bars
        height = [p.get_height() for p in plot.patches]
        ncol = int(len(height)/2)
        total = [height[i] + height[i + ncol]for i in range(ncol)] * 2
        for i, p in enumerate(plot.patches):
            plot.text(p.get_x()+p.get_width()/2,
                    height[i] + 10,
                    '{:1.0%}'.format(height[i]/total[i]),
                    ha="center", size=6) # font size looks fine when there are less than 10 categories, o.w. numbers are crunched together
        pdf_obj.savefig()
        plt.close()


def ks_new(pred, true_label, save_label, result_path, group=20, ascending=False, plot=False):
    '''
    pred:是预测label=1的概率,格式为np.array or list
    true_label:是真实的label,格式同上。
    group:分组的组数
    save_label: 存储时的文件名
    '''

    df = pd.DataFrame({'pred': pred, 'label': true_label})
    df.sort_values(by='pred', ascending=ascending, axis=0, inplace=True)  # 按照概率排序
    df.reset_index(drop=True, inplace=True)
    num = int(round(df.shape[0] * 1.0 / group, 0))
    num_bad = df.label.sum()  # label = 1的数据
    num_good = df.shape[0] - df.label.sum()
    cum_good = [0]
    cum_bad = [0]
    for i in range(1, group + 1):
        if i != group:
            cum_bad.append(df.loc[0:(i * num), 'label'].sum() * 1.0 / num_bad)
            cum_good.append((1 - df.loc[0:(i * num), 'label']).sum() * 1.0 / num_good)
        else:
            cum_bad.append(df.loc[:, 'label'].sum() * 1.0 / num_bad)
            cum_good.append((1 - df.loc[:, 'label']).sum() * 1.0 / num_good)
    ksrate = np.array(cum_bad) - np.array(cum_good)
    if plot:
        fig=plt.figure(figsize=(8, 6))
        cum_good, = plt.plot(cum_good, color='blue', marker='o', linestyle='-')
        cum_bad, = plt.plot(cum_bad, color='red', marker='o', linestyle='-')
        ks, = plt.plot(ksrate, color='SteelBlue', marker='o', linestyle='-')
        plt.legend([cum_good, cum_bad, ks], ['cum_good', 'cum_bad', 'ks'], loc=2)
        plt.axvline(pd.Series(ksrate).idxmax(), color='Black', linestyle='--')
        plt.title('%s, KS: %s ' % (save_label, ksrate.max()))

        result_path = os.path.join(result_path, 'figure/KS/')
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        plt.savefig(os.path.join(result_path, save_label + '_KS.png'), format='png', dpi=80)

        return plt
    else:
        return ksrate.max()


def ks_new_sw(pred,sample_weight,save_label,result_path,plot=False):
    '''
    pred:是预测label=1的概率,格式为np.array or list
    sample_weight(dataFrame):['sample_weight','y']
    result_path: 存储路径
    '''
    #sample_weight=pd.DataFrame(sample_weight)
    merge_data=pred.to_frame('p_train').merge(sample_weight,left_index=True,right_index=True)
    data_mid=pd.pivot_table(merge_data,index='p_train',values='sample_weight',aggfunc=np.sum).reset_index()
    data_mid.sort_values(['p_train'],ascending=False,inplace=True)
    data_mid['cum']=data_mid['sample_weight'].cumsum()
    data_mid['percent']=data_mid['cum']/data_mid['sample_weight'].sum()
    data_mid['percent']=data_mid['percent'].astype(str)
    data_mid['percent']=data_mid['percent'].apply(lambda x:x[0:4])
    list_mid=[]
    for i in range(20):
        a=round(((i+1)*0.05-0.01),2)
        list_mid.append(a)
    list_mid=pd.DataFrame(list_mid)
    list_mid.rename(columns={0:'percent'},inplace=True)
    list_mid['percent']=list_mid['percent'].astype(str)
    data_fianl=pd.merge(data_mid,list_mid,on='percent',how='inner')
    data_fianl.sort_values(['p_train'],ascending=False,inplace=True)
    data_fianl.drop_duplicates(['percent'],keep='last',inplace=True)
    bounds=list(data_fianl['p_train'])
    bounds.pop()
    boundaries=[np.inf]+bounds+[-np.inf]
    boundaries.reverse()
    merge_data['bin']=pd.cut(merge_data['p_train'],boundaries,precision=0).astype(str)
    decile_2=pd.pivot_table(merge_data,index='bin',columns='y',values='sample_weight',aggfunc=np.sum).reset_index()
    decile_2.columns=['bin','NGood','NBad']
    decile_2.sort_values(['bin'],ascending=False,inplace=True)
    decile_2['PercentBad']=decile_2['NBad']/decile_2['NBad'].sum()
    decile_2['PercentGood']=decile_2['NGood']/decile_2['NGood'].sum()
    decile_2['cumBad']=decile_2['PercentBad'].cumsum()
    decile_2['cumGood']=decile_2['PercentGood'].cumsum()
    decile_2['KS']=decile_2['cumBad']-decile_2['cumGood']
    cum_good=np.array(decile_2['cumGood'])
    cum_bad=np.array(decile_2['cumBad'])
    ksrate=np.array(decile_2['KS'])
    if plot:
        fig=plt.figure(figsize=(8, 6))
        cum_good, = plt.plot(cum_good, color='blue', marker='o', linestyle='-')
        cum_bad, = plt.plot(cum_bad, color='red', marker='o', linestyle='-')
        ks, = plt.plot(ksrate, color='SteelBlue', marker='o', linestyle='-')
        plt.legend([cum_good, cum_bad, ks], ['cum_good', 'cum_bad', 'ks'], loc=2)
        plt.axvline(pd.Series(ksrate).idxmax(), color='Black', linestyle='--')
        plt.title('%s, KS: %s ' % (save_label, ksrate.max()))

        result_path = os.path.join(result_path, 'figure/KS/')
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        plt.savefig(os.path.join(result_path, save_label + '_KS.png'), format='png', dpi=80)

        return plt
    else:
        return ksrate.max()


def ks_simplified(cum_good, cum_bad, ksrate, model_label):
    '''
    plot result from decile calculation table
    cum_good: decile 输出表里面的累积Good占比
    cum_bad: decile 输出表里面的累积Bad占比
    ksrate: decile 输出表里的KS
    '''
    fig=plt.figure(figsize=(8, 6))
    cum_good, = plt.plot(cum_good, color='red', marker='o', linestyle='-')
    cum_bad, = plt.plot(cum_bad, color='blue', marker='o', linestyle='-')
    ks, = plt.plot(ksrate, color='SteelBlue', marker='o', linestyle='-')
    plt.legend([cum_good, cum_bad, ks], ['cum_good', 'cum_bad', 'ks'], loc=2)
    plt.axvline(pd.Series(ksrate).idxmax(), color='Black', linestyle='--')
    plt.title('KS: %s for %s' % (round(ksrate.max(), 3), model_label))
    return plt




def print_AUC_one(Y_train, dtrain_predprob, save_label, result_path):
    fpr_train, tpr_train, thresholds = metrics.roc_curve(Y_train, dtrain_predprob)
    roc_auc_train = metrics.auc(fpr_train, tpr_train)
    fig=plt.figure(figsize=(8, 6))
    #rcParams['figure.figsize'] = (8, 8)
    plt.title('%s ROC' % save_label)
    plt.plot(fpr_train, tpr_train, 'b', label='AUC = %0.3f'% roc_auc_train)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1], 'k--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    result_path = os.path.join(result_path, 'figure/AUC/')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    plt.savefig(os.path.join(result_path, save_label + '_AUC.png'), format='png', dpi=80)

    #plt.close()
    return roc_auc_train,fig


def print_AUC_one_sw(dtrain_predprob, sample_weight,save_label, result_path):
    """
    plot 包含sample_weight的AUC曲线

    Args:
    """
    merge_data=dtrain_predprob.to_frame('p_train').merge(sample_weight,left_index=True,right_index=True,how='inner')
    Y_train=np.array(merge_data['y'])
    dtrain_predprob=np.array(merge_data['p_train'])
    sw=np.array(merge_data['sample_weight'])
    fpr_train, tpr_train, thresholds = metrics.roc_curve(Y_train, dtrain_predprob,sample_weight=sw)
    roc_auc_train = metrics.auc(fpr_train, tpr_train,reorder=True)
    fig=plt.figure(figsize=(8, 6))
    #rcParams['figure.figsize'] = (8, 8)
    plt.title('%s ROC' % save_label)
    plt.plot(fpr_train, tpr_train, 'b', label='AUC = %0.3f'% roc_auc_train)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1], 'k--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    result_path = os.path.join(result_path, 'figure/AUC/')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    plt.savefig(os.path.join(result_path, save_label + '_AUC.png'), format='png', dpi=80)

    #plt.close()
    return roc_auc_train,fig


def score_dist_plot(score, bins, save_label, result_path):
    dist_plot = score.plot(kind='hist', bins=bins, figsize=(15,5), alpha=0.5, rwidth=0.8)

    plt.title('%s Score Distribution' % save_label)

    result_path = os.path.join(result_path, 'figure/score_dist/')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    fig = dist_plot.get_figure()
    fig.savefig(os.path.join(result_path, save_label + '_dist_plot.png')\
                , format='png', dpi=80)
    return fig


def compare_dist(data, x_name, group, save_label, result_path, num_bins=10, cut_bounds=[]):
    """
    Compare data distribution between two or more groups. 默认将数据等分为10分。

    Args:
    data (pd.DataFrame): the data to be plotted
    x_name (str): the column name to be plotted
    group (str): the column name to identify groups
    save_label (str): identify the figure file name
    result_path (str): the path to save the figure. The plot will be saved
        under the path figure/score_dist/
    num_bins (int): 等分需要分的箱数。首选这种方式。
    cut_bounds (list): default=[]. If provided, then the data[x_name] will be
        binned accordingly. Otherwise, 默认等分方式。
    """
    data = data.copy()
    if len(cut_bounds) > 0:
        data.loc[:, x_name] = pd.cut(data.loc[:, x_name], cut_bounds)
    else:
        data.loc[:, x_name] = pd.qcut(data.loc[:, x_name], num_bins, duplicates='drop')

    plot_data = mu.crosstab_percent(data[x_name], data[group], data_format='wide')
    plot_data.plot(marker='*',figsize=(15,5))#, ylim=(0.05, 0.12)
    plt.xticks(rotation=45)
    plt.title('Distribution Comparison of %s' % x_name)
    plt.xlabel('bins')
    plt.ylabel('Percent')

    result_path = os.path.join(result_path, 'figure/compare_dist/')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, save_label + "_compare_dist.png"))
    return plt




def lasso_path(X, y, eps = 5e-3):
    """
    eps (float): the smaller it is the longer is the path
    """
    X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)
    # Compute paths
    print("Computing regularization path using the lasso...")
    alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps)

    # print("Computing regularization path using the positive lasso...")
    # alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
    #     X, y, eps, positive=True)
    # print("Computing regularization path using the elastic net...")
    # alphas_enet, coefs_enet, _ = enet_path(
    #     X, y, eps=eps, l1_ratio=0.8)
    #
    # print("Computing regularization path using the positive elastic net...")
    # alphas_positive_enet, coefs_positive_enet, _ = enet_path(
    #     X, y, eps=eps, l1_ratio=0.8, positive=True)

    # Display results
    fig=plt.figure(figsize=(9, 8))
    ax = plt.gca()
    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_alphas_lasso = -np.log10(alphas_lasso)
    for coef_l, c in zip(coefs_lasso, colors):
        l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)

    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    plt.title('Lasso Paths')
    plt.axis('tight')
    return plt

def check_badrate_trend(x_cat, y, month, result_path, normalize=False, heatmap=True):
    """
    画badrate趋势图，热图

    Args:
    x_cat (pd.Series): the binned x variable, where the index is apply_id or unique
        identifier
    y (pd.Series): the y, where the index is apply_id or unique identifier
    month (pd.Series): the month or time interval that the application belongs to,
        where the index is apply_id or unique identifier
    result_path (str): the save path of the plot. The plot will be save in
        figure/badByTime/ under the result_path passed in
    normalize (bool): default=False. If set True, the event rate will be normalized
        before plotting

    Return:
    badrate_trend (pd.DataFrame): the event rate df where index is bin of x
        , and column is month (or time interval)
    """
    data = x_cat.to_frame('x_cat').merge(month.to_frame('time_interval'), \
                                left_index=True, right_index=True)\
                          .merge(y.to_frame('Y'), left_index=True, \
                                right_index=True)
    monthly_size_pivot=pd.pivot_table(data,index='x_cat',columns='time_interval',aggfunc='size')
    key_order_size = mt.BinWoe().order_bin_output(monthly_size_pivot.reset_index(),'x_cat' ).x_cat.tolist()
    monthly_size_pivot.index=pd.Series(monthly_size_pivot.index).astype(str)
    monthly_size_pivot=monthly_size_pivot.loc[key_order_size, :]

    bin_obj = mt.BinWoe()
    monthly_woe_list = []
    for month in data.time_interval.unique():
        sub_data = data.loc[data.time_interval == month, :].copy()
        sub_woe = bin_obj.calculate_woe(sub_data.x_cat, sub_data.Y)
        sub_woe = sub_woe[['bin', 'EventRate']]
        sub_woe.loc[:, 'time_interval'] = month
        monthly_woe_list.append(sub_woe)

    monthly_woe = pd.concat(monthly_woe_list)
    monthly_woe_pivot = monthly_woe.pivot(index='bin', columns='time_interval', values='EventRate')
    # key_order = monthly_woe_pivot.index.tolist()
    # key_order.reverse()
    key_order = mt.BinWoe().order_bin_output(monthly_woe_pivot.reset_index(),'bin' ).bin.tolist()
    key_order.reverse()
    monthly_woe_pivot = monthly_woe_pivot.loc[key_order, :]
    plt.figure(figsize=(14, 10))
    #cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
    if normalize:
        #使用该时间段的整体违约率作为benchmark
        monthly_woe_pivot = (monthly_woe_pivot - data.groupby('time_interval').agg({'Y':'mean'}).Y) / (data.groupby('time_interval').agg({'Y':'mean'}).Y)
        if heatmap:
            sns.heatmap(monthly_woe_pivot, cmap = plt.cm.Blues, linewidths = 0.05, annot=True)
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.title(x_cat.name+' : Normalized Event rate by time trend')
            plt.xlabel('time interval')
            plt.ylabel(x_cat.name)
        else:
            monthly_woe_pivot.plot(figsize=(15,5))
            plt.title(x_cat.name+' : Event rate by time trend')
            plt.xlabel(x_cat.name)
    else:
        if heatmap:
            sns.heatmap(monthly_woe_pivot, cmap = plt.cm.Blues, linewidths = 0.05, annot=True)
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.title(x_cat.name+' : Event rate by time trend')
            plt.xlabel('time interval')
            plt.ylabel(x_cat.name)
        else:
            monthly_woe_pivot.plot(figsize=(15,5))
            plt.title(x_cat.name+' : Event rate by time trend')
            plt.xlabel(x_cat.name)


    result_path = os.path.join(result_path, 'figure/badByTime/')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    plt.savefig(os.path.join(result_path, x_cat.name + '.png'), format='png', dpi=80)

    return monthly_woe_pivot, monthly_size_pivot, plt



def check_badrate_trend_simplified(ready_data_to_plot, result_path, eventrate_by_time=[], heatmap=True):
    """
    画badrate趋势图，热图

    Args:
    ready_data_to_plot (pd.DataFrame): 已经按照时间维度计算好的逾期率表格。每列为时间维度
        每行为数据分箱。取值为逾期率
    result_path (str): the save path of the plot. The plot will be save in
        figure/badByTime/ under the result_path passed in
    eventrate_by_time (pd.Series): index 为时间维度名称，应与ready_data_to_plot的列名
        一致。值为当前时间维度整体的逾期率。如果传入改数据，则默认画图会经过nomalize
    heatmap (bool): default=True. If set False, 画出的图是线图

    Return:
    badrate_trend (pd.DataFrame): the event rate df where index is bin of x
        , and column is month (or time interval)
    """
    variable_name = ready_data_to_plot.index.name
    ready_data_to_plot.index = pd.Series(ready_data_to_plot.index).astype(str)
    key_order = mt.BinWoe().order_bin_output(pd.DataFrame({'bin': ready_data_to_plot.index}), 'bin').bin.tolist()
    key_order.reverse()
    ready_data_to_plot = ready_data_to_plot.loc[key_order, :]


    plt.figure(figsize=(14, 10))
    #cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
    if len(eventrate_by_time) > 0:
        #使用该时间段的整体违约率作为benchmark
        ready_data_to_plot = (ready_data_to_plot - eventrate_by_time) / eventrate_by_time
        if heatmap:
            sns.heatmap(ready_data_to_plot, cmap = plt.cm.Blues, linewidths = 0.05, annot=True)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.title(variable_name+' : Normalized Event rate by time trend')
            plt.xlabel('time interval')
            plt.ylabel(variable_name)
        else:
            ready_data_to_plot.plot(figsize=(15,5))
            plt.title(variable_name+' : Event rate by time trend')
            plt.xlabel(variable_name)
    else:
        if heatmap:
            sns.heatmap(ready_data_to_plot, cmap = plt.cm.Blues, linewidths = 0.05, annot=True)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.title(variable_name+' : Event rate by time trend')
            plt.xlabel('time interval')
            plt.ylabel(variable_name)
        else:
            ready_data_to_plot.plot(figsize=(15,5))
            plt.title(variable_name+' : Event rate by time trend')
            plt.xlabel(variable_name)


    result_path = os.path.join(result_path, 'figure/badByTime/')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    plt.savefig(os.path.join(result_path, variable_name + '.png'), format='png', dpi=80)
    plt.close()




def missing_trend(x, month, result_path=None, replace=True):
    """
    画缺失值趋势图

    Args:
    x (pd.Series): the original x variable, where the index is apply_id or unique
        identifier
    month (pd.Series): the month or time interval that the application belongs to,
        where the index is apply_id or unique identifier
    result_path (str): the save path of the plot. The plot will be save in
        figure/missingByTime/ under the result_path passed in

    Return:
    rates_by_month (pd.Series): index is time interval, series value is missing rate
    """
    data = x.to_frame('x').merge(month.to_frame('time_interval'), \
                                left_index=True, right_index=True)
    if replace:
        data = data.replace(-9999, np.nan)\
                   .replace(-8887, np.nan)\
                   .replace(-8888, np.nan)\
                   .replace('-9999', np.nan)\
                   .replace('-8887', np.nan)\
                   .replace('-8888', np.nan)
    rates_by_month = 1 - (data.groupby('time_interval').count() * 1.0).div(data.time_interval.value_counts(), 0)
    plt.figure(figsize=(14, 5))
    plt.plot(list(range(len(rates_by_month))), np.array(rates_by_month.x))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
    plt.xticks(list(range(len(rates_by_month))), rates_by_month.index.values, rotation=45)
    plt.ylim(0,1)
    plt.xlabel('on time_interval')
    plt.ylabel('missing rate')
    plt.title(x.name+' : Missing rate by time')
    if result_path is not None:
        result_path = os.path.join(result_path, 'figure/missingByTime/')
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        plt.savefig(os.path.join(result_path, x.name + '.png'), format='png', dpi=80)

    return rates_by_month, plt

def variable_psi_plot(var_psi_result, beforeName, afterName,RESULT_PATH):
    """
    画缺每个变量的PSI图

    Args:
        var_psi_result (pd.Series): 每个变量的PSI统计结果，
                                由utils3.metric.Performance().variable_psi()求得
        beforeName(str): 基准时点名称，例如：train
        afterName(str)：比较时点名称，例如：test
        RESULT_PATH (str): 图片保存路径

    Return:
        None
    """
    for i in var_psi_result[u'指标英文'].unique():
        subset = var_psi_result[var_psi_result[u'指标英文']==i][[u'基准时点占比',u'比较时点占比']]
        subset.index = var_psi_result[var_psi_result[u'指标英文']==i][u'分箱']
        subset = subset.rename(columns={u'基准时点占比':beforeName,u'比较时点占比':afterName})
        plt.figure(figsize=(14, 5))
        subset.plot(marker = '*')
        plt.title(i+" :PSI = %.4f"%var_psi_result[var_psi_result[u'指标英文']==i].PSI.mean())
        try:
            subset.index.astype(int)
        except:
            plt.xticks(rotation=45)
        plt.xlabel('bins')
        path_dic = os.path.join(RESULT_PATH, "figure",\
                            beforeName+'_'+afterName+'_vars_PSI')
        if not os.path.exists(path_dic):
            os.makedirs(path_dic)
        path = os.path.join(path_dic, i + ".png")
        plt.tight_layout()
        plt.savefig(path, format='png', dpi=80)

    return plt


def variable_badrate_compare(base_cat, base_y, compare_cat, compare_y, base_name,\
                            compare_name, var_dict, cat_encoding_map0, RESULT_PATH,\
                            verbose=True, plot=True):
    """
    对比基准样本和比较样本变量在各分箱的逾期率

    Args:
    base_cat (pd.DataFrame): 分箱好的基准样本X
    base_y (pd.Series): 基准样本的y
    compare_cat (pd.DataFrame): 分箱好的比较样本X
    compare_y (pd.Series): 比较样本的y
    var_dict (pd.DataFrame): 字典
    cat_encoding_map0 (dict): 变量分箱dictionary
    verbose (bool): defautl=True.
    """
    base_woe = mt.BinWoe().calculate_woe_all(base_cat, base_y, var_dict, cat_encoding_map0, verbose=verbose)
    compare_woe = mt.BinWoe().calculate_woe_all(compare_cat, compare_y, var_dict, cat_encoding_map0, verbose=verbose)
    merged_on_cols = ['数据源', '指标英文', '指标中文', '数据类型', '指标类型', '分箱', '分箱对应原始分类']
    selected = merged_on_cols + ['逾期率', 'N', '分布占比']
    combined = base_woe[selected].rename(columns={'逾期率': '%s逾期率'%base_name, 'N':'%s样本量'%base_name, '分布占比':'%s分布占比'%base_name})\
                                 .merge(compare_woe[selected].rename(columns={'逾期率': '%s逾期率'%compare_name, 'N':'%s样本量'%compare_name, '分布占比':'%s分布占比'%compare_name}),\
                                 on=merged_on_cols, how='left')

    if plot:
        for i in combined['指标英文'].unique():
            subset = combined[combined[u'指标英文']==i][[u'基准逾期率',u'比较逾期率']]
            subset.index = combined[combined[u'指标英文']==i][u'分箱']
            subset = subset.rename(columns={'基准逾期率':base_name, '比较逾期率':compare_name})
            plt.figure(figsize=(14, 5))
            subset.plot(marker = '*')
            plt.title('Bad Rate Comparison for %s' % i)
            try:
                subset.index.astype(int)
            except:
                plt.xticks(rotation=45)
            plt.xlabel('bins')
            path_dic = os.path.join(RESULT_PATH, "figure",\
                                base_name+'_'+compare_name+'_vars_badrate')
            if not os.path.exists(path_dic):
                os.makedirs(path_dic)
            path = os.path.join(path_dic, "%s.png" % i)
            plt.tight_layout()
            plt.savefig(path, format='png', dpi=80)
            plt.close()

    return combined


def variable_badrate_compare_sw(base_cat, base_y,base_sample_weight, compare_cat, compare_y,compare_sample_weight, base_name,\
                            compare_name, var_dict, cat_encoding_map0, RESULT_PATH,\
                            verbose=True, plot=True):
    """
    对比基准样本和比较样本变量在各分箱的逾期率

    Args:
    base_cat (pd.DataFrame): 分箱好的基准样本X
    base_y (pd.Series): 基准样本的y
    compare_cat (pd.DataFrame): 分箱好的比较样本X
    compare_y (pd.Series): 比较样本的y
    var_dict (pd.DataFrame): 字典
    cat_encoding_map0 (dict): 变量分箱dictionary
    verbose (bool): defautl=True.
    """
    base_woe = mt.BinWoe().calculate_woe_all_sw(base_cat, base_y,base_sample_weight,var_dict, cat_encoding_map0, verbose=verbose)
    compare_woe = mt.BinWoe().calculate_woe_all_sw(compare_cat, compare_y, compare_sample_weight,var_dict, cat_encoding_map0, verbose=verbose)
    merged_on_cols = ['数据源', '指标英文', '指标中文', '数据类型', '指标类型', '分箱', '分箱对应原始分类']
    selected = merged_on_cols + ['逾期率', 'N', '分布占比']
    combined = base_woe[selected].rename(columns={'逾期率': '%s逾期率'%base_name, 'N':'%s样本量'%base_name, '分布占比':'%s分布占比'%base_name})\
                                 .merge(compare_woe[selected].rename(columns={'逾期率': '%s逾期率'%compare_name, 'N':'%s样本量'%compare_name, '分布占比':'%s分布占比'%compare_name}),\
                                 on=merged_on_cols, how='left')

    if plot:
        for i in combined['指标英文'].unique():
            subset = combined[combined[u'指标英文']==i][[u'基准逾期率',u'比较逾期率']]
            subset.index = combined[combined[u'指标英文']==i][u'分箱']
            subset = subset.rename(columns={'基准逾期率':base_name, '比较逾期率':compare_name})
            plt.figure(figsize=(14, 5))
            subset.plot(marker = '*')
            plt.title('Bad Rate Comparison for %s' % i)
            try:
                subset.index.astype(int)
            except:
                plt.xticks(rotation=45)
            plt.xlabel('bins')
            path_dic = os.path.join(RESULT_PATH, "figure",\
                                base_name+'_'+compare_name+'_vars_badrate')
            if not os.path.exists(path_dic):
                os.makedirs(path_dic)
            path = os.path.join(path_dic, "%s.png" % i)
            plt.tight_layout()
            plt.savefig(path, format='png', dpi=80)
            plt.close()

    return combined



def reject_score_dis(reject_score,train_good_score,train_bad_score,RESULT_PATH):
    """
    为拒绝样本打分并与测试集中的好人和坏人进行比较

    Args:
        reject_score (pd.Series): 拒绝样本的分数
        train_good_score (pd.Series): 测试集中好人的分数
        train_bad_score (pd.Series):  测试集中坏人的分数
    Return:
        None
    """
    plt.figure(figsize=(14, 5))
    bins_ = list(np.linspace(min(train_good_score.min(),train_bad_score.min()),\
                            max(train_good_score.max(),train_bad_score.max()),11))
    all_data = pd.cut(reject_score, bins_).value_counts()/len(reject_score)
    all_data = all_data.to_frame('reject')
    all_data['good'] = pd.cut(train_good_score,bins_).value_counts()/len(train_good_score)
    all_data['bad'] = pd.cut(train_bad_score,bins_).value_counts()/len(train_bad_score)
    all_data.sort_index().plot(marker='*')
    plt.xticks(rotation=45)
    plt.title('Reject Score VS Good and Bad')
    plt.ylabel('percent')
    path_dic = os.path.join(RESULT_PATH, "figure","compare_dist")
    if not os.path.exists(path_dic):
            os.makedirs(path_dic)
    path = os.path.join(path_dic, "reject_score_dis" + ".png")
    plt.tight_layout()
    plt.savefig(path, format='png', dpi=80)
    plt.show()

def plot_score_compare(score_psi_result,beforeName,afterName,RESULT_PATH):
    """
    画缺每个变量的PSI图

    Args:
        score_psi_result (pd.Series):打分的PSI统计结果，由matrix中的Performance()类的variable_psi()求得
        beforeName(str): 基准时点名称，例如：train
        afterName(str)：比较时点名称，例如：test
        RESULT_PATH (str): 图片保存路径

    Return:
        None
    """
    subset = score_psi_result[[u'基准时点占比',u'比较时点占比']]
    subset = subset.rename(columns={u'基准时点占比':beforeName,u'比较时点占比':afterName})
    subset.index = score_psi_result[u'分箱']
    subset.plot(marker='*')
    plt.title((afterName+' VS '+ beforeName + " PSI:%.4f"%score_psi_result.PSI.mean()))
    plt.xlabel('bins')
    plt.ylabel('Percent')
    plt.xticks(rotation=45)
    path_dic = os.path.join(RESULT_PATH, "figure","compare_dist")
    if not os.path.exists(path_dic):
            os.makedirs(path_dic)
    path = os.path.join(path_dic,(afterName+' VS '+ beforeName + " PSI") + ".png")
    plt.tight_layout()
    plt.savefig(path, format='png', dpi=80)
    plt.show()


def plot_lift_curve(badcum_dict, base_label, sample_set, Y, RESULT_PATH):
    if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)

    x = np.linspace(0.1, 1, len(badcum_dict[base_label]))
    fig=plt.figure(figsize=(8, 6))
    plt.plot(x, badcum_dict[base_label], 'ko--', label=base_label)
    other_labels = [i for i in badcum_dict.keys() if i != base_label]
    # 太多线就重叠在一起不好看了
    if len(other_labels) > 7:
        raise("Out of line color&shape choices")
    line_color_choices = ['rv-.', 'bv:', 'ms-', 'gv--', 'cd-.',
                          'yo--']
    line_color_choices_map = dict(zip(other_labels, line_color_choices[:len(other_labels)]))
    for label in other_labels:
        x = np.linspace(0.1, 1, len(badcum_dict[label]))
        plt.plot(x, badcum_dict[label], line_color_choices_map[label], label=label)
    #
    plt.legend()
    plt.xlabel('% of Sample')
    plt.ylabel('% of Target')
    plt.title('Lift Curves, sample: %s, Y:%s' % (sample_set, Y))
    plt.savefig(os.path.join(RESULT_PATH, 'liftcurve_%s_%s.png' % (sample_set, Y)), format='png', dpi=80)
    plt.close()



def plot_backscore_approval_rate(data, RESULT_PATH, model_label, save_label):
    fig=plt.figure(figsize=(8, 6))
    plt.plot(data.ApprovedBadRate, data.ApprovalRate, 'ms-')
    plt.xlabel('Bad Rate(model training definition) if Approved')
    plt.ylabel('Approval Rate')
    plt.title('Approval Rate vs Bad Rate on Backscore Set for %s' % model_label)
    plt.savefig(os.path.join(RESULT_PATH, 'approval_rate_anaysis_%s.png' % save_label), format='png', dpi=80)
    plt.close()



def plot_colinearity(corr_matrix_plot, RESULT_PATH, SAVE_LABEL):
    f, ax = plt.subplots(figsize=(10, 8))

    # Diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with a color bar
    sns.heatmap(corr_matrix_plot, cmap=cmap, center=0,
                linewidths=.25, cbar_kws={"shrink": 0.6})

    # Set the ylabels
    ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
    ax.set_yticklabels(list(corr_matrix_plot.index), size = int(160 / corr_matrix_plot.shape[0]));

    # Set the xlabels
    ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
    ax.set_xticklabels(list(corr_matrix_plot.columns), size = int(160 / corr_matrix_plot.shape[1]));
    plt.title(SAVE_LABEL + ' Colinearity Heatmap', size = 14)

    RESULT_PATH = os.path.join(RESULT_PATH, 'figure', 'colinearity')
    if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)

    FILE_PATH = os.path.join(RESULT_PATH, "%s_colinearity.png" % SAVE_LABEL)
    plt.savefig(FILE_PATH, format='png', dpi=80)
    plt.close()



class BinPlotAdjustUI(object):
    """
    在UI弹框中挑选变量，并对变量画图，图为分箱分布bar chart和bad rate line chart。
    可根据图片调整分箱并重新画图。调整后的分箱spec会被记录并被输出到rebin_spec中存储。
    且最终图片会被存储。
    """
    def __init__(self, X, y, result_path, data_path, rebin_spec, var_dict):
        self.bin_obj = mt.BinWoe()
        self.root = Tk()
        self.root.title("Coarse Classing Bin Adjust and Plotting")
        self.root.geometry('600x600') # 设计窗口的大小
        self.X = X
        self.y = y
        self.RESULT_PATH = os.path.join(result_path, 'figure/distNbad/')
        self.DATA_PATH = data_path
        self.rebin_spec = rebin_spec
        self.var_dict = var_dict


    def bin_dist_bad_trend(self, x_cat, y, col_name, result_path):
        """
        画图：分箱分布bar chart和bad rate line chart

        Args:
        x_cat (pd.Series): 单一变量分箱好的数据
        y (pd.Series): y标签
        col_name (str): 指标英文名
        """
        # sns.set(style="whitegrid", palette="pastel", color_codes=True)
        self.var_woe = self.bin_obj.calculate_woe(x_cat, y)
        logging.info('var_name:%s' % list(self.var_woe['var_code'])[0])
        logging.info('Bins:%s' % list(self.var_woe['bin']))
        logging.info('IV:%s' % str(round(list(self.var_woe['IV'])[0],2)))
        logging.info('KS:%s' % str(round(list(self.var_woe['KS'])[0],2)))
        plt.figure(figsize=(7, 5))

        # distribution bar chart
        ax = self.var_woe.PercentDist.plot(kind='bar', label='Amount')
        plt.title(col_name +' IV:'+str(round(list(self.var_woe['IV'])[0],2))+ ' KS:'+str(round(list(self.var_woe['KS'])[0],2)))
        plt.xlabel('Bins')
        plt.ylabel('PercentDist')
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)

        # bad rate line chart
        aw = ax.twinx()
        plt.plot(aw.get_xticks(), self.var_woe.EventRate, 'r-*', label='bad_rate')
        aw.legend(loc='best')
        plt.ylabel('bad_rate')
        aw.yaxis.grid(False)

        # specify xticks
        plt.xticks(aw.get_xticks(), self.var_woe.bin, rotation=45)
        plt.tight_layout()

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        plt.savefig(os.path.join(result_path, col_name + '.png'), format='png', dpi=80)
        plt.close()


    def pop_plot_window(self, col_name):
        additional_plot_window = Toplevel(self.root)
        additional_plot_window.geometry('600x400')
        additional_plot_window.title('Plot Display Window')

        canvas = Canvas(additional_plot_window, bg=None, height=600, width=600)
        image = Image.open(os.path.join(self.RESULT_PATH, col_name + '.png'))
        self.image_tk_file = ImageTk.PhotoImage(image)
        image = canvas.create_image(10, 10, anchor='nw', image=self.image_tk_file)
        canvas.pack(side='top')



    def plot_selected_variable(self):
        selected = self.lbox.get(self.lbox.curselection())
        x = self.X[[selected]].copy() # selecting this way makes it a DataFrame
        rebin_spec = {k:v for k,v in list(self.rebin_spec.items()) if k in selected}
        x_cat = self.bin_obj.convert_to_category(x, self.var_dict, rebin_spec, verbose=True)
        self.bin_dist_bad_trend(x_cat[selected], self.y, selected, self.RESULT_PATH)
        self.selected_var.set(selected)
        self.pop_plot_window(selected)


    def change_bins_and_plot(self):
        selected = self.lbox.get(self.lbox.curselection())
        bin_boundaries = json.loads(self.bin_boundaries.get())
        if self.other_categories.get() == '':
            other_categories = []
        else:
            other_categories = json.loads(self.other_categories.get())
        rebin_spec = {'cut_boundaries': bin_boundaries, 'other_categories': other_categories}
        is_change = messagebox.askyesno(title='Sure to change?',
                                        message='you are going to change numerical bin bounds, do it now?')
        if is_change:
            self.rebin_spec[selected] = rebin_spec
            save_data_to_pickle(self.rebin_spec, self.DATA_PATH, 'coarse_classing_rebin_spec.pkl')
            self.plot_selected_variable()


    def change_bins_num_and_plot(self):
        selected = self.lbox.get(self.lbox.curselection())
        bin_num = int(self.bin_num.get())
        is_change = messagebox.askyesno(title='Sure to change?',
                                        message='you are going to number of auto classing bins, do it now?')
        if is_change:
            x = self.X[[selected]].copy() # selecting this way makes it a DataFrame
            x_cat, all_encoding_map, all_spec_dict = self.bin_obj.binning(x, self.y, self.var_dict, num_max_bins=bin_num)
            woe_iv_df = self.bin_obj.calculate_woe_all(x_cat, self.y, self.var_dict, all_spec_dict)
            rebin_spec = self.bin_obj.create_rebin_spec(woe_iv_df, all_spec_dict, all_encoding_map)
            self.rebin_spec[selected] = rebin_spec[selected]
            save_data_to_pickle(self.rebin_spec, self.DATA_PATH, 'coarse_classing_rebin_spec.pkl')
            self.bin_dist_bad_trend(x_cat[selected], self.y, selected, self.RESULT_PATH)
            self.selected_var.set(selected)
            self.pop_plot_window(selected)


    def rebin_varchar_variable(self):
        selected = self.lbox.get(self.lbox.curselection())
        cat_rebin = json.loads(self.cat_rebin.get())
        is_change = messagebox.askyesno(title='Sure to change?',
                                        message='you are going to change category groups, do it now?')

        if is_change:
            new_rebin = {}
            for new_label, old_labels in list(cat_rebin.items()):
                new_values = []
                for i in old_labels:
                    new_values.extend(self.rebin_spec[selected][i])

                new_rebin[int(new_label)] = new_values

            self.rebin_spec[selected] = new_rebin
            save_data_to_pickle(self.rebin_spec, self.DATA_PATH, 'coarse_classing_rebin_spec.pkl')
            self.plot_selected_variable()





    def main_ui(self, columns_to_coarse):
        self.selected_var = StringVar()
        self.var_selection_label = Label(self.root, bg='yellow', width=34, textvariable=self.selected_var)
        self.var_selection_label.pack()

        variables_to_coarse = StringVar(value=tuple(columns_to_coarse))
        self.lbox = Listbox(self.root, listvariable=variables_to_coarse, width=60)
        self.lbox.pack()

        # plot button
        Button(self.root, text='Plot', width=15, height=2, command=self.plot_selected_variable)\
            .pack()

        # 改变分类变量的分箱
        Label(self.root, width=27, text='Rebin categorical variable: ')\
           .place(x=0, y=250)
        self.cat_rebin = ttk.Entry(self.root, show=None, width=32)
        self.cat_rebin.place(x=220, y=250)
        Label(self.root, width=50, text='example: {"1":[2,3], "2":[1,4,5]}')\
           .place(x=100, y=275)

        Button(self.root, text='change category groups', width=18, height=2, command=self.rebin_varchar_variable)\
           .place(x=220, y=295)




        # 输入新分箱的输入框
        Label(self.root, width=23, text='Bin cut point:')\
           .place(x=0, y=345)
        self.bin_boundaries = ttk.Entry(self.root, show=None, width=40)
        self.bin_boundaries.place(x=150, y=345)
        Label(self.root, width=50, text='example: [-8887, 3, 4, Infinity]')\
           .place(x=100, y=370)

        Label(self.root, width=23, text='Special Categories: ')\
           .place(x=0, y=390)
        self.other_categories = ttk.Entry(self.root, show=None, width=38)
        self.other_categories.place(x=170, y=390)
        Label(self.root, width=50, text='example: [-8887, -9999, -8888]')\
           .place(x=100, y=415)


        # 改变分箱的button
        Button(self.root, text='change bin bounds', width=15, height=2, command=self.change_bins_and_plot)\
           .place(x=220, y=435)


        # 改变auto classing的分箱数
        Label(self.root, width=30, text='Change # of auto classing: ')\
           .place(x=0, y=485)
        self.bin_num = ttk.Entry(self.root, show=None, width=32)
        self.bin_num.place(x=225, y=485)
        Label(self.root, width=50, text='example: 5')\
           .place(x=100, y=510)

        Button(self.root, text='change bin #', width=15, height=2, command=self.change_bins_num_and_plot)\
           .place(x=220, y=530)


        self.root.mainloop()


class BinPlotAdjustUI_sw(object):
    """
    在UI弹框中挑选变量，并对变量画图，图为分箱分布bar chart和bad rate line chart。
    可根据图片调整分箱并重新画图。调整后的分箱spec会被记录并被输出到rebin_spec中存储。
    且最终图片会被存储。
    """
    # modify by jzw
    def __init__(self, X, y,sample_weight, result_path, data_path, rebin_spec, var_dict):
        self.bin_obj = mt.BinWoe()
        self.root = Tk()
        self.root.title("Coarse Classing Bin Adjust and Plotting")
        self.root.geometry('600x600') # 设计窗口的大小
        self.X = X
        self.y = y
        self.RESULT_PATH = os.path.join(result_path, 'figure/distNbad/')
        self.DATA_PATH = data_path
        self.rebin_spec = rebin_spec
        self.var_dict = var_dict
        self.sample_weight=sample_weight


    def bin_dist_bad_trend_sw(self, x_cat, y, col_name,sample_weight, result_path):
        """
        画图：分箱分布bar chart和bad rate line chart

        Args:
        x_cat (pd.Series): 单一变量分箱好的数据
        y (pd.Series): y标签
        sample_weight(pd.Series):样本权重
        col_name (str): 指标英文名
        """
        # sns.set(style="whitegrid", palette="pastel", color_codes=True)
        # modify by jzw
        self.var_woe = self.bin_obj.calculate_woe_sw(x_cat,sample_weight,y,col_name,ks_order='eventrate_order')
        logging.info('var_name:%s' % list(self.var_woe['var_code'])[0])
        logging.info('Bins:%s' % list(self.var_woe['bin']))
        logging.info('IV:%s' % str(round(list(self.var_woe['IV'])[0],2)))
        logging.info('KS:%s' % str(round(list(self.var_woe['KS'])[0],2)))
        plt.figure(figsize=(7, 5))

        # distribution bar chart
        ax = self.var_woe.PercentDist.plot(kind='bar', label='Amount')
        plt.title(col_name +' IV:'+str(round(list(self.var_woe['IV'])[0],2))+ ' KS:'+str(round(list(self.var_woe['KS'])[0],2)))
        plt.xlabel('Bins')
        plt.ylabel('PercentDist')
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)

        # bad rate line chart
        aw = ax.twinx()
        plt.plot(aw.get_xticks(), self.var_woe.EventRate, 'r-*', label='bad_rate')
        aw.legend(loc='best')
        plt.ylabel('bad_rate')
        aw.yaxis.grid(False)

        # specify xticks
        plt.xticks(aw.get_xticks(), self.var_woe.bin, rotation=45)
        plt.tight_layout()

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        plt.savefig(os.path.join(result_path, col_name + '.png'), format='png', dpi=80)
        plt.close()


    def pop_plot_window(self, col_name):
        additional_plot_window = Toplevel(self.root)
        additional_plot_window.geometry('600x400')
        additional_plot_window.title('Plot Display Window')

        canvas = Canvas(additional_plot_window, bg=None, height=600, width=600)
        image = Image.open(os.path.join(self.RESULT_PATH, col_name + '.png'))
        self.image_tk_file = ImageTk.PhotoImage(image)
        image = canvas.create_image(10, 10, anchor='nw', image=self.image_tk_file)
        canvas.pack(side='top')



    def plot_selected_variable(self):
        selected = self.lbox.get(self.lbox.curselection())
        x = self.X[[selected]].copy() # selecting this way makes it a DataFrame
        rebin_spec = {k:v for k,v in list(self.rebin_spec.items()) if k in selected}
        x_cat = self.bin_obj.convert_to_category(x, self.var_dict, rebin_spec, verbose=True)
        self.bin_dist_bad_trend_sw(x_cat[selected], self.y, selected,self.sample_weight, self.RESULT_PATH)
        self.selected_var.set(selected)
        self.pop_plot_window(selected)


    def change_bins_and_plot(self):
        selected = self.lbox.get(self.lbox.curselection())
        bin_boundaries = json.loads(self.bin_boundaries.get())
        if self.other_categories.get() == '':
            other_categories = []
        else:
            other_categories = json.loads(self.other_categories.get())
        rebin_spec = {'cut_boundaries': bin_boundaries, 'other_categories': other_categories}
        is_change = messagebox.askyesno(title='Sure to change?',
                                        message='you are going to change numerical bin bounds, do it now?')
        if is_change:
            self.rebin_spec[selected] = rebin_spec
            save_data_to_pickle(self.rebin_spec, self.DATA_PATH, 'coarse_classing_rebin_spec.pkl')
            self.plot_selected_variable()


    def change_bins_num_and_plot_sw(self):
        selected = self.lbox.get(self.lbox.curselection())
        bin_num = int(self.bin_num.get())
        is_change = messagebox.askyesno(title='Sure to change?',
                                        message='you are going to number of auto classing bins, do it now?')
        if is_change:
            x = self.X[[selected]].copy() # selecting this way makes it a DataFrame
            x_cat, all_encoding_map, all_spec_dict = self.bin_obj.binning(x, self.y, self.var_dict, num_max_bins=bin_num)
            woe_iv_df = self.bin_obj.calculate_woe_all_sw(x_cat, self.y,self.sample_weight, self.var_dict, all_spec_dict)
            rebin_spec = self.bin_obj.create_rebin_spec(woe_iv_df, all_spec_dict, all_encoding_map)
            self.rebin_spec[selected] = rebin_spec[selected]
            save_data_to_pickle(self.rebin_spec, self.DATA_PATH, 'coarse_classing_rebin_spec.pkl')
            self.bin_dist_bad_trend_sw(x_cat[selected], self.y, selected,self.sample_weight, self.RESULT_PATH)
            self.selected_var.set(selected)
            self.pop_plot_window(selected)


    def rebin_varchar_variable(self):
        selected = self.lbox.get(self.lbox.curselection())
        cat_rebin = json.loads(self.cat_rebin.get())
        is_change = messagebox.askyesno(title='Sure to change?',
                                        message='you are going to change category groups, do it now?')

        if is_change:
            new_rebin = {}
            for new_label, old_labels in list(cat_rebin.items()):
                new_values = []
                for i in old_labels:
                    new_values.extend(self.rebin_spec[selected][i])

                new_rebin[int(new_label)] = new_values

            self.rebin_spec[selected] = new_rebin
            save_data_to_pickle(self.rebin_spec, self.DATA_PATH, 'coarse_classing_rebin_spec.pkl')
            self.plot_selected_variable()





    def main_ui(self, columns_to_coarse):
        self.selected_var = StringVar()
        self.var_selection_label = Label(self.root, bg='yellow', width=34, textvariable=self.selected_var)
        self.var_selection_label.pack()

        variables_to_coarse = StringVar(value=tuple(columns_to_coarse))
        self.lbox = Listbox(self.root, listvariable=variables_to_coarse, width=60)
        self.lbox.pack()

        # plot button
        Button(self.root, text='Plot', width=15, height=2, command=self.plot_selected_variable)\
            .pack()

        # 改变分类变量的分箱
        Label(self.root, width=27, text='Rebin categorical variable: ')\
           .place(x=0, y=250)
        self.cat_rebin = ttk.Entry(self.root, show=None, width=32)
        self.cat_rebin.place(x=220, y=250)
        Label(self.root, width=50, text='example: {"1":[2,3], "2":[1,4,5]}')\
           .place(x=100, y=275)

        Button(self.root, text='change category groups', width=18, height=2, command=self.rebin_varchar_variable)\
           .place(x=220, y=295)




        # 输入新分箱的输入框
        Label(self.root, width=23, text='Bin cut point:')\
           .place(x=0, y=345)
        self.bin_boundaries = ttk.Entry(self.root, show=None, width=40)
        self.bin_boundaries.place(x=150, y=345)
        Label(self.root, width=50, text='example: [-8887, 3, 4, Infinity]')\
           .place(x=100, y=370)

        Label(self.root, width=23, text='Special Categories: ')\
           .place(x=0, y=390)
        self.other_categories = ttk.Entry(self.root, show=None, width=38)
        self.other_categories.place(x=170, y=390)
        Label(self.root, width=50, text='example: [-8887, -9999, -8888]')\
           .place(x=100, y=415)


        # 改变分箱的button
        Button(self.root, text='change bin bounds', width=15, height=2, command=self.change_bins_and_plot)\
           .place(x=220, y=435)


        # 改变auto classing的分箱数
        Label(self.root, width=30, text='Change # of auto classing: ')\
           .place(x=0, y=485)
        self.bin_num = ttk.Entry(self.root, show=None, width=32)
        self.bin_num.place(x=225, y=485)
        Label(self.root, width=50, text='example: 5')\
           .place(x=100, y=510)

        Button(self.root, text='change bin #', width=15, height=2, command=self.change_bins_num_and_plot_sw)\
           .place(x=220, y=530)


        self.root.mainloop()
