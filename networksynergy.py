from pandas import DataFrame, Series
from collections import defaultdict
import community
import pandas as pd; import numpy as np
import csv #使用内置的CSV库
import os
import matplotlib.pyplot as plt
import networkx as nx

os.chdir("/Users/eric/Documents/statwork/NetworkSynergy/NetSynergyChange")
cwd = os.getcwd()

"""第二阶段：将配对后的重构信息转换格式导入网络图"""
"""将包含历史信息的数据处理，将档期OFDI与历史OFDI配对"""
treateddf = pd.read_stata('NetsynergyInformation.dta', index_col = 'groupkeys')

temporatyofdis = treateddf[['siren', 'year', 'paysnew', 'paysold']]

temporatyofdis = temporatyofdis.drop_duplicates()
temporatyofdis.head()

temporatyofdis[-10:]

"""采用遍历各分组的方式可以简单地将数据框按照分组变量拆分"""
samples = [group for (_, _, _), group in df.groupby(['siren', 'year', 'paysnew'])]

"""
"""这里曾出现DUG：当只有一条信息时，重新索引会将一个元素变成Series，因此会产生不同的格式"""
"""构造函数：拆分数据"""

def split_by_index(index):
    
    temps = temporatyofdis.loc[index]
    if type(temps) == pd.core.frame.DataFrame:
        return temps
    else:
        return DataFrame(temps).T

    
"""使用函数将其拆分"""
indexlst = list(np.unique(temporatyofdis.index))
samples = [split_by_index(index) for index in indexlst]
"""

"""将每一个DataFrame转换成列表与元组"""
def transforinfomation(df):
        
    lst = df.iloc[0, :3].values
    results=list(lst)
    results.append(list(zip(df['paysold'], df['paysnew'])))
       
    return results


networkchangesinfo = [transforinfomation(df) for df in samples]


locations = pd.read_stata('all_lat_lang.dta')

'''输出各节点结构洞与中心度指数-list'''
'''注意的是：由于OFDI网络协同效应（Network Synergy）重构了贸易网络，因此采用有向无权网络'''
def NetworkChanges(changes):
    df = pd.read_stata("./tradefiles/wtf_bilat_exptop3_%s.dta" % changes[1])
    '''生成网络图'''    
    '''获取边'''
    f = df.iloc[:, 2]
    to = df.iloc[:, 3]
    
    edgesList = list(zip(f, to))
    
    nodelist = list(locations.iloc[:,6])
    
    ''''''
    '''构建有向图'''
    G = nx.DiGraph()
    G.add_nodes_from(nodelist)
    """添加有向有权网络图的边"""
    G.add_edges_from(edgesList)
    """加入Network Synergy重构"""
    G.add_edges_from(changes[3])

    """获取网络指标并保存到数组Array中"""
    data = np.array([changes[0]], dtype = object)
    data = np.vstack((data, [changes[1]]))
    data = np.vstack((data, [changes[2]]))
    """获取网络指标"""
    constraintdict = nx.constraint(G)
    data = np.vstack((data, [constraintdict[changes[2]]]))
    try:
        """"""
        eigenvector_centrality = nx.eigenvector_centrality(G)
        data = np.vstack((data, [eigenvector_centrality[changes[2]]]))
        
    except nx.exception.PowerIterationFailedConvergence as e:
        """如果报错，就将其特征向量中心度用缺失值填充"""
        data = np.vstack((data, np.nan))


    return data
    
"""构建一个更改数据格式的函数，以方便能够成功导出"""
def DtaFrameshr(dta):
    dta.columns = ['siren', 'year', 'paysnum', 'Structural_Holes_Post',
                          'Eigenvector_Centrality_Post']
    """将Object对象转换格式"""
    dta['Structural_Holes_Post'] = dta['Structural_Holes_Post'].astype(float)
    dta['Eigenvector_Centrality_Post'] = dta['Eigenvector_Centrality_Post'].astype(float)
    return dta
    
import datetime
lst = [['siren'], ['year'],['paysnum'],[100],[200]]
dfs = np.array(lst, dtype = object)
"""循环"""
print(datetime.datetime.now())

counts = 0
for row in networkchangesinfo:
    dfs = np.hstack((dfs,NetworkChanges(row)))
    counts += 1
    if (counts%5000) == 0 :
        print(datetime.datetime.now())
        print("现在循环到：%d/46171......" % counts)
            
synergychanges = DataFrame(dfs)
synergychanges = synergychanges.T
dtaframes = DtaFrameshr(synergychanges)
dtaframes.to_stata("NSCdf.dta", write_index = False)
