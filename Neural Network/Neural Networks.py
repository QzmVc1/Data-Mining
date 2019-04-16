import numpy as np
from scipy.special import expit
import pandas as pd
import random

class neuralNetwork():
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learningrate=0.6):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.wih = np.random.normal(0.0,pow(self.hidden_nodes,-0.5),(self.hidden_nodes,self.input_nodes))
        self.who = np.random.normal(0.0,pow(self.output_nodes,-0.5),(self.output_nodes,self.hidden_nodes))
        self.lr = learningrate
        self.activation_function = lambda x:expit(x)

    def trainNet(self,input_data,target_data):
        # print(self.wih)
        # print('*'*40)
        # print(input_data)
        # print('*' * 40)
        # print(target_data)
        # print('*' * 40)
        hidden_input = np.dot(self.wih, input_data)
        hidden_output = self.activation_function(hidden_input)
        final_input = np.dot(self.who,hidden_output)
        final_output = self.activation_function(final_input)

        final_error = target_data - final_output
        hidden_error = np.dot(self.who.T,final_error)

        self.who += self.lr * np.dot(final_error * final_output * (1-final_output),hidden_output.T)
        self.wih += self.lr * np.dot(hidden_error * hidden_output * (1-hidden_output),input_data.T)

    def query(self,input_data):
        hidden_output = self.activation_function(np.dot(self.wih,input_data))
        output = self.activation_function(np.dot(self.who,hidden_output))
        return output

def handleData(lst):
    for i in range(len(lst)):
        if lst[i] == 'setosa':
            lst[i] = random.uniform(0,1/3)
        elif lst[i] == 'versicolor':
            lst[i] = random.uniform(1/3,2/3)
        else:
            lst[i] = random.uniform(2/3,1)
    return np.array(lst)

def convertToCategory(Ndarray):
    lst = list(Ndarray.flatten())
    for i in range(len(lst)):
        if lst[i] >= 0 and lst[i] <1/3:
            lst[i] = 0
        elif lst[i] >= 1/3 and lst[i] < 2/3:
            lst[i] = 1
        else:
            lst[i] = 2
    return np.array(lst)

def successRate(data,target):
    return np.sum(((data==target)==True))

if __name__ == '__main__':
    input_data = pd.read_csv(r'C:\Users\QzmVc1\Desktop\Learning Materials\Data_Mining\iris.csv',index_col=0)
    input_data = np.array(input_data.sample(frac=1).reset_index(drop=True)).T
    train_data = np.array(input_data[0:4,:70],dtype="float")
    test_data = np.array(input_data[0:4,70:],dtype="float")
    target = handleData(list(input_data[4,70:]))


n = neuralNetwork(4,90,1,0.004)
for k in range(70):
    n.trainNet(test_data,target)
    # print(n.query(train_data))
    # print(convertToCategory(n.query(train_data)))
    # print(convertToCategory(target))

cmp1 = convertToCategory(n.query(test_data))
cmp2 = convertToCategory(target)
print("成功率为：%.2f%%" % successRate(cmp1,cmp2))