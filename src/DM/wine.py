import numpy as np
import operator
import random
import math
import csv

#def splitdata(data,split):
#    trainingSet = []
#    testSet = []
#    print(data)
#    a = random.shuffle(data)
#    trainingSet = a[:50]
#    testSet = a[50:]
#    return trainingSet , testSet


def euclideanDistance(dataset,testdata):
    a=0
    count = 1
    for x in range(len(testdata)):
        if count != len(testdata):
            a += np.power((float(dataset[x]) - float(testdata[x])),2)
        else :
            count += 1
    return math.sqrt(a)


def getNeighbors(trainingSet, testInstance , k ):
    #print(trainingSet)
    #print(testInstance)
    dis = {}
    for x in range(len(trainingSet)):
        #print(trainingSet[x])
        dist = euclideanDistance(trainingSet[x],testInstance)
        dis[str(trainingSet[x])] = dist
    #print(dis)
    distt = sorted(dis.items(), key=operator.itemgetter(1), reverse=False)
  
    neighbors = []
    for x in range(k):
        neighbors.append(distt[x][0])
                
    #print(neighbors)
    return neighbors


#training1 = [[1,2,3,4],[2,1,1,4],[5,5,4,7],[1,2,5,4]]
#test1 = [1,2,5,4]
#k =2
#neighbors = getNeighbors(training1, test1, k)
#print(neighbors)

def getmaxmin(data):
    mdict = []
    for cloumn in range(len(data[0])):
        cloumndata = []
        for row in data:
            cloumndata.append(row[cloumn])
        #print(cloumndata)
        mdict.append(max(cloumndata))
        
    mdict[0] = 1
        #mdict['attribute'+str(cloumn+1)].append(min(cloumndata))
    #print(mdict)
    return mdict


def getResponse(neighbors):
    classvotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][1]
        if response in classvotes:
            classvotes [response] += 1
        else :
            classvotes [response] =  1
    #print(classvotes)
    sortedvotes = sorted(classvotes.items(), key=operator.itemgetter(1), reverse = True)
    #print(sortedvotes[0][0])
    return sortedvotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    f_predictions = [float(i)for i in predictions]
    for x in range(len(testSet)):
        if testSet[x][0] == f_predictions[x]:
            correct +=1
        else :
            print('mismatch : testSet is {}  prediction is {}'.format(testSet[x][0],f_predictions[x]))
    return (correct/float(len(testSet)))*100.0

def main():
    f = open('inputfile.csv', 'r',encoding='utf-8')
    data = []
    n_data = []
    raw_data = []
    rawdict=[]
    for row in csv.reader(f):
        data.append(row)
    f.close()
    #print(data)
    mdict = getmaxmin(data)
    for x in range (len(data)):
        rawdict.append(1)

    #print(data)
    #print(mdict)
    #len(data) 178 
    #(len(data[0]) 14
        
    for i in range(len(data)):
        n_data.append([float(x) / float(y) for x, y in zip(data[i], mdict)])
        raw_data.append([float(x) / float(y) for x, y in zip(data[i], rawdict)])
    print(raw_data)
    
    #for column in range (len(data[0])):
    #    columndata = []
    #    for row in data:
    #        columndata = columndata.append(row[column]/mdict[column])
    #    print(columndata)
    
    #print(columndata)
    #[x/y for x,y in zip(mdict,columndata)]
    random.shuffle(n_data)
    trainingSet = n_data[:89]
    testSet = n_data[89:]
    #print(data)
    #print(n_data)
    predictions=[]
    k = 3
    for x in range (len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy'+repr(accuracy)+'%')


for i in range(100):
    main()


