from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = [[1.0 , 1.0],
               [1.0 , 2.0],
               [1.0 , 3.0],
               [1.0 , 4.0],
               [1.0 , 5.0],
               [1.0 , 6.0]
        ]
    labelMat = [3.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
        ]
    return dataMat, labelMat

def standRegress(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx)==0.0:
        print "This matrix is singular, cannnot do inverse"
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

if __name__ =='__main__':
    xArr, yArr = loadDataSet()
    print xArr,yArr
    ws = standRegress(xArr, yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(mat(xArr)[:,1].flatten().A[0], mat(yArr).T[:,0].flatten().A[0])
    xCopy = mat(xArr).copy()
    xCopy.sort(0)
    yHat = xCopy*ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()
    print corrcoef(yHat.T , mat(yArr))    #输出相关系数
