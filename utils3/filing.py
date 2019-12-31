import os
import xlwings as xw
import pandas as pd

class ModelSummary2Excel(object):
    '''
    重写XlModelSummary，更加通用
    汇总所有统计结果的类
    使用过程：实例化一个XlModelSummary()类，并调用run()方法

    构造函数：
    Args:
        result_path：excel的输出目录
        fig_path：KS、AUC、score_dis图片的存储主目录，也就是到figure目录即可
        file_name：统计结果的名称，例如：“summary.xlsx”
        data_dic:所有数据的汇总字典，详情见Demo

    Returns：
        excel：所有统计结果的汇总
    '''
    def __init__(self, result_path, fig_path, file_name, data_dic):
        self.app = xw.App(visible=False,add_book=False)
        self.result_path = result_path
        self.fig_path = fig_path
        self.file_name = file_name
        self.data_dic = data_dic
        self.wb = self.app.books.add()


    # 直接跑出结果
    def run(self):
        #主要内容
        for sheetname, data_to_add in self.data_dic.items():
            if isinstance(data_to_add, dict):
                self.add_sheet_dict(sheetname, data_to_add)
            elif isinstance(data_to_add, list):
                self.add_sheet_list(sheetname, data_to_add)
            elif isinstance(data_to_add, pd.DataFrame):
                if '评分卡' in sheetname:
                    self.add_score_card(sheetname, data_to_add)
                else:
                    self.add_sheet(sheetname, data_to_add)

        #目录
        try:
            sheet = self.wb.sheets['Sheet1']
            sheet.name = 'summary'
            sheet.activate()
            self.table_format(sheet.range('B2'), [[x.name] for x in self.wb.sheets])
        except:
            try:
                sheet = self.wb.sheets['工作表1']
                sheet.name = 'summary'
                sheet.activate()
                self.table_format(sheet.range('B2'), [[x.name] for x in self.wb.sheets])
            except:
                pass
                
        #保存
        self.wb.save(os.path.join(self.result_path,self.file_name))
        #退出
        # self.app.quit()

    def add_sheet(self, sheet_name, data=None):
        """
        添加一个单独DataFrame到sheet中

        Args:
        sheet_name (str): sheet名称
        data(pd.DataFrame): 默认为空，可以自己选择添加一个sheet页

        Returns:
        None
        """
        sheet = self.wb.sheets.add(after=self.wb.sheets.count)
        sheet.name = sheet_name
        rng = sheet.range('A1')
        rng.options(index=False).value = data
        sheet.range('1:3').autofit()
        #sheet.range('1:1').color=(152,245,255)
        rng.expand().color=(240, 255, 255)
        sheet.range((1,1),(1,rng.end('right').column)).color=(135, 206, 235)


    def add_sheet_dict(self, sheet_name, data_dict={}):
        """
        添加一个单独DataFrame到sheet中

        Args:
        sheet_name (str): sheet名称
        data_dict(dict): 字典value包含pd.DataFrame或者图片路径下的地址（fig_path下一层的完整路径），字典的key将作为区域的名称

        Returns:
        None
        """
        sheet = self.wb.sheets.add(sheet_name,after=self.wb.sheets.count)
        i = 1
        for name, data in data_dict.items():
            if ('_picture' in name) or (isinstance(data, str)):
                pic_height=400
                sheet.range('A'+str(i)).value = name.replace('_picture', '')
                sheet.pictures.add(top=sheet.range('A1').height*i,left=sheet.range('A1').width*1\
                                ,height=pic_height ,image=os.path.join(self.fig_path,data))
                i = i + int(pic_height/16) + 3
            else:
                sheet.range('A'+str(i)).value = name
                rng = sheet.range('B'+str(i+1))
                self.table_format(rng, data, autofit=True)
                i = i + len(data) + 3
        sheet.range('1:3').autofit()


    def add_sheet_list(self, sheet_name, data_list=[]):
        """
        添加list of DataFrame到sheet中

        Args:
        sheet_name (str): sheet名称
        data_list(list): data_list值全部为pd.DataFrame

        Returns:
        None
        """
        sheet = self.wb.sheets.add(sheet_name,after=self.wb.sheets.count)
        i = 1
        j = 1
        for data in data_list:
            if isinstance(data, pd.DataFrame):
                sheet.range('A'+str(i)).value = 'data table {}:'.format(j)
                rng = sheet.range('B'+str(i+1))
                self.table_format(rng, data, autofit=True)
                i = i + len(data) + 3
                j = j + 1
            else:
                print('list data {} is not pd.DataFrame'.format(j))
        sheet.range('1:3').autofit()



    def add_score_card(self, sheet_name, score_card):
         """
         添加评分卡sheet。因为添加scorecard和普通的一个dataframe不一样，要在评分卡旁边
         添加评分卡转换的基本信息。所以单独写一个

         Args:
         sheet_name (str): sheet名称
         score_card (pd.DataFrame): 程序生成的评分卡文件。不是最后的部署文档评分卡文件

         Returns:
         None
         """
         sheet = self.wb.sheets.add()
         sheet.name = sheet_name
         sheet.range('A3').options(index=False).value = score_card
         side_notes = ['intercept','PDO','Base Points','Base Odds','B','A','score =',\
                  '部署注意：','最大分','最小分']
         intercept_score = score_card.loc[score_card['指标英文']=='intercept','变量打分'].item()
         score_max = score_card.groupby('指标英文')['变量打分'].max().sum()
         score_min = score_card.groupby('指标英文')['变量打分'].min().sum()
         side_notes_dict = {
         '截距分': intercept_score,
         'PDO': 75,
         'Base Points': 660,
         'Base Odds': '0.067, bad:good = 1:15',
         'B': 108.202,
         'A': 366.983,
         'score =': 'A-B*log(odds)',
         '最大分': score_max,
         '最小分': score_min
         }
         side_notes_df = pd.Series(side_notes_dict).to_frame()
         col_position = score_card.shape[1] + 3
         sheet.range((3, col_position)).options(index=False).value = side_notes_df



    def table_format(self, rng, data=None, autofit=True):
        """
        添加一个单独DataFrame到sheet中指定的单元格位置，并设置颜色和自动宽度

        Args:
        rng (str): 区域
        data(pd.DataFrame): DataFrame

        Returns:
        None
        """
        sheet = rng.sheet
        rng.options(index=False).value = data
        if autofit:
            rng.expand().autofit()
        #表格颜色
        rng.expand().color=(240, 255, 255)
        #表头颜色
        sheet.range((rng.row,rng.column), (rng.row, rng.column+rng.expand().columns.count-1)).color=(135, 206, 235)
