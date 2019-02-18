import numpy as np
import pandas as pd
import random

""" 
函数功能：计算香农熵 
参数说明： 
    dataSet：原始数据集
返回： 
    ent:香农熵的值
"""
def calEnt(dataSet):
    n = dataSet.shape[0]                                    # 获取行数
    # type(class_count) = Series
    class_count = dataSet.iloc[:,-1].value_counts()         # 计算最后一列 各类别分别有多少个
    p = class_count / n                                     # 计算各分类的概率
    ent = (-p * np.log2(p)).sum()                           # 计算香浓熵
    return ent

""" 
函数功能：根据信息增益选择出最佳数据集切分的列 
参数说明： 
    dataSet：原始数据集
返回： 
    axis:数据集最佳切分列的索引
"""

def bestSplit(dataSet):
    baseEnt = calEnt(dataSet)                                      # 计算原始熵
    bestGainRate = -999                                                # 初始化信息增益
    axis = -1                                                      # 初始化最佳特征索引
    for i in range(dataSet.shape[1]-1):                            # 对所有特征列进行遍历
        levels = dataSet.iloc[:,i].value_counts().index            # 获取该特征中的所有取值
        series_ent = 0                                             # 初始化该特征的信息熵
        for j in levels:                                           # 对该特征中的所有可能取值进行遍历
            childSet = dataSet[dataSet.iloc[:,i]==j]               # 以特征中的所有取值划分子数据集
            ent = calEnt(childSet)
            series_ent += (childSet.shape[0]/dataSet.shape[0])*ent # 计算信息熵
        infoGain = baseEnt - series_ent                            # 信息增益
        p1 = (dataSet.iloc[:,i].value_counts())/(dataSet.shape[0])
        HA = (-p1 * np.log2(p1)).sum()
        infoGainRate = infoGain / HA                               # 计算信息增益率
        if infoGainRate > bestGainRate:                                # 选取最大信息增益率
            bestGainRate = infoGainRate
            axis = i
    return axis                                                    # 返回最大信息增益所在列的索引

def mySplit(dataSet,axis,value):
    col = dataSet.columns[axis]
    redataSet = dataSet.loc[dataSet[col]==value,:].drop(col,axis=1)
    return redataSet
""" 
函数功能：基于最大信息增益切分数据集，递归构建决策树 
参数说明： 
    dataSet：原始数据集（最后一列是标签）
返回： 
    myTree：字典形式的树
"""
def createTree(dataSet):
    featlist = list(dataSet.columns)                                        # 存储特征值
    classlist = dataSet.iloc[:,-1].value_counts()
    if classlist[0] == dataSet.shape[0] or dataSet.shape[1]==1:             # 递归终止条件
        return classlist.index[0]
    axis = bestSplit(dataSet)                                               # 确定出当前最佳切分列的索引
    bestfeat = featlist[axis]                                               # 获取该索引对应的特征
    myTree = {bestfeat:{}}                                                  # 采用字典嵌套的方式存储树信息
    del featlist[axis]                                                      # 删除当前特征
    valuelist = set(dataSet.iloc[:,axis])                                   # 提取最佳切分列所有属性值
    for value in valuelist:                                                 # 对每一个属性值递归建树
        myTree[bestfeat][value] = createTree(mySplit(dataSet,axis,value))
    return myTree

""" 
函数功能：对一个测试实例进行分类 
参数说明： 
    inputTree：已经生成的决策树 
    labels：存储选择的最优特征标签 
    testVec：测试数据列表，顺序对应原数据集
返回： 
    classLabel：分类结果
"""
def classify(inputTree,labels,testVec):
    firstStr = next(iter(inputTree))                                       # 获取决策树的第一个节点
    secondDict = inputTree[firstStr]                                       # 下一个字典
    featIndex = labels.index(firstStr)                                     # 第一个节点所在列的索引
    for key in secondDict.keys():                                          # 进行分支
        if (testVec[featIndex] == key):
            if type(secondDict[key]) == dict:
                classLabel = classify(secondDict[key], labels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel
""" 
函数功能：对测试集进行预测，并返回预测后的结果 
参数说明： 
    train：训练集 
    test：测试集
返回： 
    test：预测好分类的测试集
"""
def acc_classify(train,test):
    inputTree = createTree(train)
    labels = list(train.columns)
    result = []
    for i in range(test.shape[0]):
        testVec = test.iloc[i,:-1]
        classLabel = classify(inputTree,labels,testVec)
        result.append(classLabel)

    df = test.copy()
    df['predict'] = result
    acc = (df.iloc[:,-1]==df.iloc[:,-2]).mean()
    print(f"模型预测准确率为{acc}")                         # Python3.6 新增
    return df

if __name__ == '__main__':
    dataSet = pd.read_csv("buy_computers.csv")
    # 打乱数据集
    dataSet = dataSet.sample(frac=1).reset_index(drop=True)
    df = acc_classify(dataSet.iloc[0:6, :], dataSet.iloc[7:, :])
    print(df)

