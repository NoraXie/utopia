 # -- coding: UTF-8 --
from flask import render_template, flash, redirect, url_for, session, request, g
from app import app
from app.forms import LoginForm, NameForm
from datetime import datetime
from flask_moment import Moment
import sys
import pandas as pd
import numpy as np
from models import User
from jinja2 import Template
from config import Sql_templates, model_dict, models_config, user_config
from decorators import login_required
from utils3.data_io_utils import *
reload(sys)
sys.setdefaultencoding('utf-8')

sql_PSI = Sql_templates['PSI']
sql_KS = Sql_templates['KS']
sql_PSI_with_product = Sql_templates['PSI_with_product']
sql_KS_with_product = Sql_templates['KS_with_product']

@app.context_processor
def my_context_processer():
    username = session.get('username')
    if username:
        user = User(username)
        return {'user':user}
    return {}

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    summary_data = pd.read_excel('app/monitor_summary.xlsx')
    summary_data_recent1 = summary_data.sort_values(by=[u'监控日期',u'模型编码'],ascending=False)\
    .drop_duplicates(u'模型编码',keep='first')
    summary_data_recent1.fillna('',inplace=True)
    return render_template('index.html',model_dict=model_dict,summary_data_recent1=summary_data_recent1.as_matrix(), models_config=models_config)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method=='POST':
        if request.form['email'].split('@')[0] in user_config.keys() \
        and request.form['password']== user_config.get(request.form['email'].split('@')[0])['password']:
            session['username']=request.form['email'].split('@')[0]
            return redirect(url_for('index'))
        else:
            flash(u'用户名或密码错误')
            return render_template('login.html')
    else:
        return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/psi')
@login_required
def psi():
    model_name = session.get('model_name')

    # test_data = presto_read_sql_df(sql_test)
    '''
    test_data = pd.read_excel('/Users/pintec/Desktop/psi_data_test.xlsx')
    test_data = test_data[test_data[u'model_name']==model_name]
    '''
    if len(model_name.split('@')) != 1:
        test_data = presto_read_sql_df(Template(sql_PSI_with_product).render(model_name=model_name.split('@')[0],product_name=model_name.split('@')[1]))
    else:
        test_data = presto_read_sql_df(Template(sql_PSI).render(model_name=model_name))
    data_all = test_data.pivot_table(index=['model_var_name'],columns='dt',values='psi_index',aggfunc=sum).fillna(0)
    var_data_all = test_data.pivot_table(index=['model_var_name','model_var_cat'],columns='dt',values='curr_dist').reset_index().fillna(0)
    default_var_dist = test_data.pivot_table(index=['model_var_name','model_var_cat'],columns='dt',values='default_dist').reset_index().fillna(0)
    default_var_dist = default_var_dist[list(default_var_dist.columns)[:3]]
    default_var_dist.columns = ['model_var_name','model_var_cat','default_dist']
    var_data_all = pd.merge(default_var_dist,var_data_all,on=['model_var_name','model_var_cat'])
    var_data_all = pd.concat([var_data_all[var_data_all['model_var_name']=='模型分'],var_data_all[var_data_all['model_var_name']!='模型分']])
    score_nums = test_data[test_data[u'model_var_name']==u'模型分']\
    .pivot_table(index=['model_var_name'],columns='dt',values='score_num',aggfunc=sum).fillna(0).as_matrix()[0]

    session['recent_PSI'] = data_all[data_all.index==u'模型分'].as_matrix()[0][-1]
    session['recent_PSI_num'] =  score_nums[-1]
    time_window = test_data.groupby('dt')[['ob_start_dt','ob_end_dt']].max().values[-1]
    session['recent_PSI_window'] = time_window[0]+'~'+time_window[1]

    # 统计所有变量PSI
    names = list(data_all.index)
    dates = [str(i)[:10] for i in list(data_all.columns)]
    values = data_all.as_matrix()
    details = []
    for i,j in zip(names,values):
        details.append({'name':i,'data':list(j)})

    # 统计所有分箱数据
    var_data_all_summary = []
    for i in var_data_all[u'model_var_name'].unique():
        tmp_detail = []
        for j in var_data_all.columns[2:]:
            tmp_detail.append({'detail_name':str(j),'detail_data':list(var_data_all[var_data_all[u'model_var_name'] == i][j])})
        tmp = {'title':i,'bins':list(var_data_all[var_data_all[u'model_var_name'] == i][u'model_var_cat']),'data':tmp_detail}
        var_data_all_summary.append(tmp)
    return render_template('psi.html',names=names,dates = dates,details = details,\
    var_data_all_summary=var_data_all_summary,score_nums=score_nums,monitor_var='PSI',model_name=model_name,model_dict=model_dict)

@app.route('/ks')
@login_required
def ks():
    model_name = session.get('model_name')

    '''
    test_data = pd.read_excel('/Users/pintec/Desktop/ks_data_test.xlsx')
    test_data = test_data[test_data[u'model_name']==model_name]
    '''
    if len(model_name.split('@')) != 1:
        test_data = presto_read_sql_df(Template(sql_KS_with_product).render(model_name=model_name.split('@')[0],product_name=model_name.split('@')[1]))
    else:
        test_data = presto_read_sql_df(Template(sql_KS).render(model_name=model_name))

    if test_data.shape[0]==0:
        return render_template('blank.html',model_name=model_name,model_dict=model_dict,monitor_var='KS')
    else:
        data_all = test_data.pivot_table(index=['model_var_name'],columns='dt',values='ks',aggfunc=max).fillna(0)
        var_data_all = test_data.pivot_table(index=['model_var_name','model_var_cat'],columns='dt',values='event_rate').reset_index().fillna(0)
        default_var_dist = test_data.pivot_table(index=['model_var_name','model_var_cat'],columns='dt',values='default_event_rate').reset_index().fillna(0)
        default_var_dist = default_var_dist[list(default_var_dist.columns)[:3]]
        default_var_dist.columns = ['model_var_name','model_var_cat','default_event_rate']
        var_data_all = pd.merge(default_var_dist,var_data_all,on=['model_var_name','model_var_cat'])
        var_data_all = pd.concat([var_data_all[var_data_all['model_var_name']=='模型分'],var_data_all[var_data_all['model_var_name']!='模型分']])
        approved_nums = test_data[test_data[u'model_var_name']==u'模型分']\
        .pivot_table(index=['model_var_name'],columns='dt',values='approve_num',aggfunc=sum).fillna(0).as_matrix()[0]

        session['recent_KS'] =  data_all[data_all.index==u'模型分'].as_matrix()[0][-1]
        session['recent_KS_num'] =  approved_nums[-1]

        time_window = test_data.groupby('dt')[['ob_start_dt','ob_end_dt']].max().values[-1]
        session['recent_KS_window'] = time_window[0]+'~'+time_window[1]
        # 统计所有变量PSI
        names = list(data_all.index)
        dates = [str(i)[:10] for i in list(data_all.columns)]
        values = data_all.as_matrix()
        details = []
        for i,j in zip(names,values):
            details.append({'name':i,'data':list(j)})

        # 统计所有分箱数据
        var_data_all_summary = []
        for i in var_data_all[u'model_var_name'].unique():
            tmp_detail = []
            for j in var_data_all.columns[2:]:
                tmp_detail.append({'detail_name':str(j),'detail_data':list(var_data_all[var_data_all[u'model_var_name'] == i][j])})
            tmp = {'title':i,'bins':list(var_data_all[var_data_all[u'model_var_name'] == i][u'model_var_cat']),'data':tmp_detail}
            var_data_all_summary.append(tmp)
        return render_template('ks.html',names=names,dates = dates,details = details\
        ,var_data_all_summary=var_data_all_summary,approved_nums=approved_nums,monitor_var='KS',model_name=model_name,model_dict=model_dict)

@app.route('/monitor_summary',methods=['GET', 'POST'])
@login_required
def monitor_summary():
    if request.args.get("model_name"):
        session['model_name'] = request.args.get("model_name")
        session['recent_PSI'] = ''
        session['recent_PSI_num'] = ''
        session['recent_PSI_window'] = ''
        session['recent_KS'] = ''
        session['recent_KS_num'] = ''
        session['recent_KS_window'] = ''
    else:
        pass
    model_name = session.get('model_name')
    summary_data = pd.read_excel('app/monitor_summary.xlsx')
    summary_data = summary_data.fillna('')
    try:
        summary_data_part = summary_data[summary_data[u'模型编码']==model_name]
        summary_data_part.sort_index(by=u'监控日期',ascending=False,inplace=True)
        summary_data_part[u'监控日期'] = summary_data_part[u'监控日期'].apply(lambda x : str(x)[:10])
        summary_data_matrix = summary_data_part.as_matrix()
    except:
        summary_data_matrix = [['' for i in range(summary_data.shape[1])]]
    if request.method=='GET':
        return render_template('monitor_summary.html',summary_data = summary_data_matrix\
        ,monitor_var='summary',model_name=session['model_name'],model_dict=model_dict,models_config = models_config)
    else:
        if session.get('username') in models_config[model_name][u'owner'].split(',') or session.get('username')=='admin':
            summary_data = summary_data[~((summary_data[u'监控日期'] == request.form['date'])\
            &(summary_data[u'模型编码'] == model_name))]
            summary_data.to_excel('app/monitor_summary.xlsx')
            return redirect(url_for('monitor_summary'))
        else:
            return render_template('blank.html',model_name=model_name,model_dict=model_dict,monitor_var='summary')


@app.route('/edit',methods=['GET', 'POST'])
@login_required
def edit():
    model_name = session.get('model_name')
    recent_PSI = session.get('recent_PSI')
    recent_PSI_num = session.get('recent_PSI_num')
    recent_PSI_window = session.get('recent_PSI_window')
    recent_KS = session.get('recent_KS')
    recent_KS_num = session.get('recent_KS_num')
    recent_KS_window = session.get('recent_KS_window')
    if session.get('username') in models_config[model_name][u'owner'].split(',') or session.get('username')=='admin':
        if request.method=='GET':
            return render_template('edit.html',monitor_var='edit',model_name=model_name,model_dict=model_dict,\
            recent_PSI=recent_PSI,recent_PSI_num=recent_PSI_num,recent_KS=recent_KS,recent_KS_num=recent_KS_num,\
            recent_KS_window=recent_KS_window,recent_PSI_window = recent_PSI_window)
        else:
            summary_data = pd.read_excel('app/monitor_summary.xlsx')
            now = datetime.strftime(datetime.now(),'%Y-%m-%d')
            summary = request.form['summary']
            try:
                PSINum = float(request.form['PSINum'])
            except:
                PSINum = np.nan
            try:
                PSIValue = float(request.form['PSIValue'])
            except:
                PSIValue = np.nan
            try:
                KSNum = float(request.form['KSNum'])
            except:
                KSNum = np.nan
            try:
                KSValue = float(request.form['KSValue'])
            except:
                KSValue = np.nan
            group = request.form['group']
            all_summary = [now, PSINum,PSIValue,KSNum,KSValue,summary,group,model_name,models_config[model_name]['product_line']]
            summary_data.reset_index(drop=True,inplace=True)
            summary_data.loc[len(summary_data)] = all_summary
            summary_data.to_excel('app/monitor_summary.xlsx')
            return redirect(url_for('monitor_summary'))
    else:
        return render_template('blank.html',model_name=model_name,model_dict=model_dict,monitor_var='edit')
