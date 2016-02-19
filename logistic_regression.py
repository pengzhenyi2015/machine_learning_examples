import math
from numpy import *
import operator
import matplotlib.pyplot as plt
import matplotlib

#用于训练的数据集
def loadDataSet():
    dataMat=[[1.0, 1.0, 1.0],
             [1.0, 2.0, 2.0],
             [1.0, 2.0, 1.0],
             [1.0, 3.0, 2.0],
             [1.0, 3.0, 1.0],
             [1.0, 4.0, 1.0],
             [1.0, 4.0, 2.0],
             [1.0, 2.0, 3.0],
             [1.0, 2.0, 4.0],
             [1.0, 2.0, 5.0],
             [1.0, 3.0, 4.0],
             [1.0, 3.0, 5.0],
             [1.0, 4.0, 5.0],
             [1.0, 5.0, 5.0]
             ]
    labelMat = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

#使用全局方式的逻辑回归
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 50000
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose()*error
    return weights

#随机方式的逻辑回归，优点是不用每次都计算全部的样本，因此对在线分类比较理想
#这是一种比较简单的方式，算法顺序读取样本并计算
def stocGradAscent0(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones((n,1))
    for j in range(500):
        for i in range(m):
            h = sigmoid(sum(dataMatrix[i] * weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i].transpose()
    return weights

#真正的随机读取样本并计算，而且步长alpha的值会随着训练的进行而逐渐变小，优点是可以防止抖动
def stocGradAscent1(dataMatIn, classLabels, numIter = 500):
    dataMatrix = mat(dataMatIn)
    m,n = shape(dataMatrix)
    weights = ones((n, 1))
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex].transpose()
            del(dataIndex[randIndex])
    return weights

if __name__ == '__main__':
    dataSet, labelMat = loadDataSet()
    #weights = gradAscent(dataSet, labelMat)
    weights = stocGradAscent0(dataSet, labelMat)
    dataArr = array(dataSet)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    print weights
    x = arange(1.0, 5.5, 0.1)
    y = zeros(x.size)
    #y = (-weights[0]-weights[1]*x)/weights[2]
    for i in range(x.size):
        y[i] = (-weights[0] - weights[1]*x[i])/weights[2]
    print x, y
    ax.plot(x,y)
    plt.xlabel("x"); plt.ylabel("y")
    plt.show()
