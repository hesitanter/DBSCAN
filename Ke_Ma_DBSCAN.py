# ESE590
# Ke Ma

# Data set chosen: Iris
# DBSCAN implementation


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial


class ProcessData_1():
    def __init__(self, path):
        self.data = pd.read_table(path,sep='\t',header=None)     #读入txt文件，分隔符为\t
        self.data = self.data.values.tolist()
    def process(self):
        for i in range(len(self.data)):
            self.data[i] = self.data[i][0].split(',')
            if (self.data[i][4] == 'Iris-setosa') :
                self.data[i][4] = 1
            elif (self.data[i][4] == 'Iris-versicolor') :
                self.data[i][4] = 2
            else:
                self.data[i][4]= 3
            self.data[i] = [float(x) for x in self.data[i]]
        x = np.array(self.data) #最后一列是标签1,2,3
        return x


class MyDBSCAN():
    def __init__(self, data, DistanceMatrix, eps, MinPts):
        self.data = data
        self.DM = DistanceMatrix
        self.eps = eps
        self.MinPts = MinPts
        self.cluster_num = 0
        self.num = len(self.data)
        self.label = np.zeros(self.num)  
    def DBSCAN(self):
        for index in range(self.num):
            #如果访问过，则跳过
            if (self.label[index] != 0):
                continue
            #没访问过
            neighbour = self.Find_neighbours(index)
            if (len(neighbour) >= self.MinPts):
                self.cluster_num += 1
                self.ExpandCluster(neighbour, index)
            else:
                self.label[index] = -1

    def Find_neighbours(self, index_num):
        neibour = np.where(self.DM[index_num] < self.eps)[0]
        return neibour.tolist()

    def ExpandCluster(self, neighbour, index):
        self.label[index] = self.cluster_num
        while(len(neighbour) > 0):
            index = neighbour[0]
            if (self.label[index] == -1):
                self.label[index] = self.cluster_num
            elif (self.label[index] == 0) :
                self.label[index] = self.cluster_num
                neib = self.Find_neighbours(index)
                if (len(neib) >= self.MinPts):
                    for p in range(len(neib)): #如果没被访问过才能添加进队列
                        if (self.label[p] == 0):
                            neighbour.append(p)
            neighbour.pop(0)



if __name__ == "__main__":
    obj = ProcessData_1('C:\\Users\\Ke Ma\\Desktop\\590\Week1_lab/iris.data')
    data_labeled = obj.process()
    data = data_labeled[:, 0:4]
    DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data, 'euclidean'))
    dbscan = MyDBSCAN(data, DistanceMatrix, 2, 2) # eps, Min points
    dbscan.DBSCAN()
    print("Ideal cluster id of each node:")
    a = []
    for i in range(50):
        a.append(1)
    for i in range(50):
        a.append(2)
    for i in range(50):
        a.append(3)
    a = np.array(a)
    print(a, "\n")
    print("Cluster id after DBSCAN: ")
    print(dbscan.label)






