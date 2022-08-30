# -*- coding: utf-8 -*-

import csv
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import metrics
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import operator
global f 
f = open('output.txt', 'w')
d_list=[]


def loadcsvdata(path):
    f = open(path, 'r',encoding='utf-8')
    data = []
    for row in csv.reader(f):
        data.append(row)
    f.close()
    return data
def processlosevalue(data):
    # imp = Imputer(missing_values='?', strategy='mean', axis=0)
    # imp.fit(data)

    for i in range(len(data)):
        for j in range(len(data[i])):
            if(data[i][j]=="?"):
                data[i][j] = np.nan
    np.asarray(data)

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(data)
    # print(imp.transform(data))
    y = imp.transform(data)

    return np.array(y,"int")


def gettargetandattr(data):
    target = []
    for row in data:
    #    target.append(row[-1])
    #    row.pop(-1)
        if int(row[-1]) > 1:
            target.append(1)
            row.pop(-1)
        else:
            target.append(row[-1])
            row.pop(-1)
    #for r in target:
    #    if int(r) > 1 :
    #        r = 1
    return target
def getmaxmin(data):
    mdict = {}
    for cloumn in range(len(data[0])):
        cloumndata = []
        for row in data:
            cloumndata.append(row[cloumn])
        #mdict['attribute'+str(cloumn+1)] = []
        #mdict['attribute'+str(cloumn+1)].append(max(cloumndata))
        #mdict['attribute'+str(cloumn+1)].append(min(cloumndata))
    return mdict
def entropy(mdict,count):
    m_entropy = 0
    for r in mdict:
        pk = mdict[r] / count
        m_entropy += -(pk * np.log(pk))
    print('m_entropy',m_entropy)
def s_entropy(mdict,count):
    m_entropy = 0
    for r in mdict:
        pk = mdict[r] / count
        m_entropy += -(pk * np.log(pk))
    return m_entropy
def showdata_entropy(dimension,data):
    datalist = []
    for row in data:
        datalist.append(row[dimension])
    mdict = {}
    count = 0
    for r in datalist:
        if str(r) in mdict.keys():
            count+=1
            mdict[str(r)] = mdict[str(r)]+1
        else:
            mdict[str(r)] = 1
            count+=1
    print(mdict,count)
    weightsum = 0
    for r in mdict:
        weightsum += (int(r)/ len(data))

    return s_entropy(mdict,count), weightsum,datalist


def listattr(data):
    mdict = {}
    count = 0
    for r in data:
        if str(r) in mdict.keys():
            count+=1
            mdict[str(r)] = mdict[str(r)]+1
        else:
            mdict[str(r)] = 1
            count+=1
    print(mdict)
    entropy(mdict,count)
    # return mdict
'''get the specified column's number appearance time and mean,stard,max,min'''
def printdetail(datalist,attributelist):
    idict = {}
    for r in attributelist:
        dlist = []
        for data in datalist:
            dlist.append(data[r])
        m = np.mean(dlist)
        stard = np.std(dlist)
        m_min = np.min(dlist)
        m_max = np.max(dlist)
        listattr(dlist)
        idict[r] = []
        idict[r].append(m)
        idict[r].append(stard)
        idict[r].append(m_min)
        idict[r].append(m_max)



    return idict
'''change the value'''
def change(mean,sig,value):
    x = mean - value
    return int(x / sig) + 10
'''cml=C13choose5 mdict=get details of 0,3,4,7columns and change the values'''
def choosedimension(datalist):
    attributesize = len(datalist[0])
    candiatelist = []
    for i in range(attributesize):
        candiatelist.append(i)

    cml = []
    cml = list(combinations(candiatelist, 5))
    print(cml)
    mdict = {}
    mdict = printdetail(datalist,[0,3,4,7])
    print(mdict)


    #for r in mdict:
    #    for row in datalist:
    #        row[r] = change(mdict[r][0], mdict[r][1], row[r])
    #mdict = printdetail(datalist,[0,3,4,7])
    #print(mdict)


def printpredictresult(clf,X_test,y_test):
    #print("result:\n%s\n" % (
    #    metrics.classification_report(
    #        y_test,
    #        clf.predict(X_test))))
    a = metrics.confusion_matrix( y_test,clf.predict(X_test))
    print("confusion_matrix:\n%s\n" % (
        metrics.confusion_matrix(
            y_test,
            clf.predict(X_test))))
    print('diagonal sum:'+str(np.sum(np.diagonal(a))))
    d_list.append(np.sum(np.diagonal(a)))
    global f
    
    f.write(np.array2string(confusion_matrix(y_test, clf.predict(X_test)), separator=', ')+'\n\n\n\n')
    f.write(str(np.sum(np.diagonal(a)))+'\n\n')
    #print("Classification Report (training data):")
    #print(classification_report(y_test, clf.predict(X_test)))
    #print("Confusion Matrix (training data):")
    #conf_mtx=confusion_matrix(y_test, clf.predict(X_test))
    #print(conf_mtx)
    #plt.matshow(conf_mtx, cmap=plt.cm.gray) #plot confusion matrix
    #plt.show()

    #plt.show()
    
def predict(data,target):
    data = np.array(data,"int")


    #normalization
    #min_max_scaler = preprocessing.MinMaxScaler()
    #data1 = min_max_scaler.fit_transform(data)
    #data = np.array(data,"float32")
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33)
    clf1 = GaussianNB()
    clf1.fit(X_train, y_train)
    # print('GaussianNB    min_max_scaler')
    #printpredictresult(clf1, X_test, y_test)
    ###MultinomialNB
    #clf1 = MultinomialNB()
    #clf1.fit(X_train, y_train)
    #decision tree
    #clf1 = tree.DecisionTreeClassifier()
    #clf1.fit(X_train, y_train)
    #f.write('GaussianNB'+'\n')
    printpredictresult(clf1, X_test, y_test)
    #f.write('DecisionTree'+'\n')
    #printpredictresult(clf2, X_test, y_test)


    #
    # X_normalized = preprocessing.normalize(data, norm='l2')
    # X_train, X_test, y_train, y_test = train_test_split(X_normalized, target, test_size=0.3)
    # clf = GaussianNB()
    # clf.fit(X_train, y_train)
    # print('GaussianNB    normalize')
    # printpredictresult(clf, X_test, y_test)
    
def classdict(target):
    mdict = {}
    count = 0
    for i in range(len(target)):
        if str(target[i]) in mdict.keys():
            count+=1
            mdict[target[i]].append(i)
        else:
            mdict[target[i]] = [i]

    print('target',mdict)
    return mdict

def entropy_all(mdict,target_size):
    ent = 0
    for r in mdict:
        pk = len(mdict[r]) / target_size
        ent += -(pk * np.log(pk))
    return ent

def weightsumandentropy(dimension,data):
    datalist = []
    for row in data:
        datalist.append(row[dimension])
    mdict = {}
    count = 0
    for r in datalist:
        if r in mdict.keys():
            count+=1
            mdict[r] = mdict[r]+1
        else:
            mdict[r] = 1
            count+=1
    weightsum = 0
    for r in mdict:
        weightsum += (r/len(data))
    m_entropy = 0
    for r in mdict:
        pk = mdict[r] / len(data)
        m_entropy += -(pk * np.log(pk))

    print(weightsum)
    print('m_entropy',m_entropy)
    return weightsum*m_entropy

def calculateinformationgain(data,target):
    mclassdict = {}
    mclassdict = classdict(target)
    s = entropy_all(mclassdict,len(target))
    print("entropy_sss",s)
    len_data = len(data[0])
    for i in range(len_data):
        informationgain = s - weightsumandentropy(i,data)
        print('informationgain +',i,'dim',informationgain)

def braudratechoose(data,target):
    attributesize = len(data[0])
    mclassdict = {}
    mclassdict = classdict(target)
    for r in mclassdict:
        cml = []
        datatarget = []
        for i in mclassdict[r]:
            datatarget.append(data[i])
        for i in range(attributesize):
            ss = showdata_entropy(i,datatarget)[0]
            print('dimemsion',i,'entropy',ss)
            if ss <= 1:
                cml.append(i)
            else:
                pass
        print('candidate',cml)
    s = entropy_all(mclassdict,len(target))
    print("entropy_sss",s)

    for i in range(50):
        ncml = [1,5,6,7,8,9,10,12]
        dddlist = []
        for row in data:
            mdata = []
            for dd in ncml:
                mdata.append(row[dd])
            dddlist.append(mdata)
    
        predict(dddlist, target)

    #candiatelist = []
    #for i in range(attributesize):
    #    candiatelist.append(i)
    #global cml
    #cml = []
    #cml = list(combinations(candiatelist, 8))
    #for r in cml:
    #    print(r)
    #    f.write(str(r)+'\n\n')
    #    datalist = []
    #    for row in data:
    #        mdata = []
    #        for dd in r:
    #           mdata.append(row[dd])
    #        datalist.append(mdata)
    #    #listattr(datalist)
    #    predict(datalist, target)
    """
    ttt
    """
    # cml = []
    # for i in range(attributesize):
    #     ss = showdata_entropy(i,data)
    #     if ss <= 1.1:
    #
    #         print('dimemsion',i,'entropy',ss)
    #         cml.append(i)
    #     else:
    #         pass
    # print('candidate',cml)
    # datalist = []
    # for row in data:
    #     mdata = []
    #     for dd in cml:
    #         mdata.append(row[dd])
    #     datalist.append(mdata)
    #
    # predict(datalist, target)




data = []
data = loadcsvdata('D:/workspace/DataMining/src/DM/cleveland.csv')
print(getmaxmin(data))
target = []
target = gettargetandattr(data)
print(target)
data = processlosevalue(data)
#choosedimension(data)
output=braudratechoose(data,target)

print(np.reshape(d_list, (10, 5)))
print('d_list length ='+str(len(d_list)))
index, value = max(enumerate(d_list), key=operator.itemgetter(1))
print('max relative index = '+str(index),'diagonal_sum = '+str(value))
print(np.mean(d_list))
#print('max candidate = '+str(cml[index]))

#calculateinformationgain(data,target)
#print(data)
#print(getmaxmin(data))
#print(showdata_entropy(0, data))
# predict(data, target)
