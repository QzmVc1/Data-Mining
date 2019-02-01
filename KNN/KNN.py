import csv
import random
# 读取数据集
with open('iris.csv','r') as fp:
    reader = csv.DictReader(fp)
    datas = [row for row in reader]

# 对数据分成训练集和测试集
random.shuffle(datas)
n = len(datas) // 3
test_set = datas[0:n]
train_set = datas[n:]

# 计算欧几里得距离
def distance(d1,d2):
    res = 0
    for key in ("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"):
        res += (float(d1[key])-float(d2[key]))**2
    return res**0.5

K = 5
def KNN(data):
    # 1.计算距离
    res = [
        {'result':train['Species'],'distance':distance(data,train)}
        for train in train_set
    ]

    # 2.升序排序
    res = sorted(res,key=lambda item:item['distance'])

    # 3.取前K个数据
    res2 = res[0:K]

    # 4.加权平均
    result = {'versicolor':0,'setosa':0}
    sum = 0
    for r in res2:
        sum += r['distance']
    for r in res2:
        result[r['result']] += 1-r['distance']/sum

    if result['versicolor'] > result['setosa']:
        return 'versicolor'
    else:
        return 'setosa'


# 测试阶段
correct = 0
for test in test_set:
    result1 = test['Species']
    result2 = KNN(test)
    if result1 == result2:
        correct += 1

print("{:.2f}%".format(correct*100/len(test_set)))
