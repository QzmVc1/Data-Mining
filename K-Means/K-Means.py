"""
author: QzmVc1
Time: 2019/1/23
"""
import numpy as np
import matplotlib.pyplot as plt

# 计算欧几里得距离
def distEclud(x,y):
    return np.sqrt(np.sum((x-y)**2))

# 随机生成k个初始聚类中心
def randCent(dataSet,k):
    lst = []   # 防止生成重复
    i = 0
    m,n = dataSet.shape
    centroids = np.zeros((k,n))  # centroids 是保存k个质心的ndarray
    while i < k:
        index = np.random.randint(0,m)
        if index not in lst:
            lst.append(index)
            centroids[i,:] = dataSet[index,:]
            i += 1
    return centroids

# K-Means算法具体实现
def KMeans(dataSet,k):
    m = dataSet.shape[0]
    # 生成k个初始聚类中心
    centroids = randCent(dataSet,k)
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.zeros((m,2))  # clusterAssment数组将各点和质心关联起来
    clusterChange = True  # 循环条件
    while clusterChange:
        clusterChange = False
        for i in range(m):   # 遍历每一个点
            minIndex = -1
            minDist = 10000
            for j in range(k):  # 计算该点归属于哪一个质心
                dist = distEclud(centroids[j, :], dataSet[i, :])
                if dist < minDist:
                    minIndex = j
                    minDist = dist
            # 如果存在一个点没有收敛，则继续循环
            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                clusterAssment[i,:] = minIndex,minDist**2

        # 计算新的k个质心
        for j in range(k):
            selectrow = dataSet[np.nonzero(clusterAssment[:,0]==j)[0]]  # 获取簇类所有的点
            centroids[j,:] = np.mean(selectrow,axis=0)  # 对矩阵的行求均值

    return centroids,clusterAssment

# 画聚类图
def showCluster(dataSet,k,centroids,clusterAssment):
    m,n = dataSet.shape
    marker = ['or', 'ob', 'og', 'ok', 'Dr', 'Db', 'Dg', 'Dk', '<r', 'pr']
    if n != 2:
        print("数据不是二维的！")
    elif k > len(marker):
        print("聚类结果超出限制！")
    else:
        # 绘制所有样本
        for i in range(m):
            plt.plot(dataSet[i,0],dataSet[i,1],marker[int(clusterAssment[i,0])])
        # 绘制质心
        for i in range(k):
            plt.plot(centroids[i,0],centroids[i,1],marker[i+k])
        plt.show()


# 加载数据集
dataSet = np.loadtxt('K-Means_Data.txt')
# 聚类个数
k = 4
centroids,clusterAssment = KMeans(dataSet,k)
showCluster(dataSet,k,centroids,clusterAssment)