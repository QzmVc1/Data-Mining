import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
函数功能：创建实验数据集
参数说明：无参数
返回：
    postingList：切分好的样本词条
    classVec：类标签向量
"""
def loadDataSet():
    dataSet=[['my', 'dog', 'has', 'flea', 'problems','help','please'],
             ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
             ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
             ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
             ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
             ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']] # 切分好的词条
    classVec = [0,1,0,1,0,1]  # 类别标签向量，1代表侮辱性词汇，0代表非侮辱性词汇
    return dataSet,classVec

"""
函数功能：将切分的样本词条整理成词汇表（不重复）
参数说明：
    dataSet：切分好的样本词条
返回：
    vocabList：不重复的词汇表
"""
def createVocabList(dataSet):
    vocabSet = set()                    # 创建一个空的集合
    for doc in dataSet:                 # 遍历dataSet中的每一条言论
        vocabSet = vocabSet | set(doc)  # 取并集
    return list(vocabSet)

"""
函数功能：
根据vocabList词汇表，将inputSet向量化，向量的每个元素为 1或 0
参数说明：
    vocabList：词汇表
    inputSet：切分好的词条列表中的一条
返回：
    returnVec：文档向量,词集模型
"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)                # 创建一个其中所含元素都为0的向量
    for word in inputSet:                           # 遍历每个词条
        if word in vocabList:                       # 如果词条存在于词汇表中，则变为1
            returnVec[vocabList.index(word)] = 1
        else:
            print(" %s is not in my Vocabulary!" % word )
    return returnVec                                # 返回文档向量

"""
函数功能：生成训练集向量列表
参数说明：
    dataSet：切分好的样本词条
返回：
    trainMat：所有的词条向量组成的列表
"""
def get_trainMat(dataSet):
    trainMat = []                                          # 初始化向量列表
    vocabList = createVocabList(dataSet)                   # 生成词汇表
    for inputSet in dataSet:                               # 遍历样本词条中的每一条样本
        returnVec = setOfWords2Vec(vocabList, inputSet)    # 将当前词条向量化
        trainMat.append(returnVec)                         # 追加到向量列表中
    return trainMat


def trainNB(trainMat,classVec):
    n = len(trainMat)                          # 计算训练的文档数目
    m = len(trainMat[0])                       # 计算每篇文档的词条数
    pAb = sum(classVec)/n                      # 文档属于侮辱类的概率
    p0Num = np.ones(m); p1Num = np.ones(m)   # 词条出现数初始化为1
    p0Denom = 2; p1Denom = 2                   # 分母初始化为2
    for i in range(n):                         # 遍历每一个文档
        if classVec[i] == 1:                   # 统计属于侮辱类的条件概率所需的数据
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:                                  # 统计属于非侮辱类的条件概率所需的数据
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1V = np.log(p1Num / p1Denom)
    p0V = np.log(p0Num / p0Denom)
    return p0V,p1V,pAb                         # 返回属于非侮辱类,侮辱类和文档属于侮辱类的概率

def classifyNB(vec2Classify, p0V, p1V, pAb):
    p1 = sum(vec2Classify * p1V) + np.log(pAb)
    p0 = sum(vec2Classify * p0V) + np.log(1- pAb)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB(testVec):
    dataSet,classVec = loadDataSet()
    vocabList = createVocabList(dataSet)
    trainMat= get_trainMat(dataSet)
    p0V,p1V,pAb = trainNB(trainMat,classVec)
    thisone = setOfWords2Vec(vocabList, testVec)
    if classifyNB(thisone,p0V,p1V,pAb):
        print(testVec,'属于侮辱类')
    else:
        print(testVec,'属于非侮辱类')

#测试样本1
testVec1 = ['love', 'my', 'dalmation']
testingNB(testVec1)
#测试样本2
testVec2 = ['stupid', 'garbage']
testingNB(testVec2)