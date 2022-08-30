from sklearn.datasets import load_wine
from sklearn import metrics
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import multivariate_normal

import numpy as np
import sklearn
import matplotlib.pyplot as plt

class Map_classifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):

	def __init__(self,X,y):
		pass
	def learn(self,X,y):
		self.C_list = sorted(list(set(y)))
		x_dict   = defaultdict(int)
		x_dattolab_dict = defaultdict(lambda: {Ci: 0 for Ci in self.C_list})
		x_amount      = defaultdict(lambda: defaultdict(lambda: 0))
		x_label_amount   = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
		
		for feature, label in zip(X, y):
			Ci = label
			for i, j in enumerate(feature):
				x_amount[i][j]        += 1
				x_label_amount[i][Ci][j] += 1
				x_dict[i] += 1
				x_dattolab_dict[i][Ci] += 1
				
		for i in x_amount.keys():
			for j in x_amount[i].keys():
				#print(x_amount[i][j])
				#print(x_dict[i])
				x_amount[i][j] /= x_dict[i]
		#print(x_amount)
			

		for i in x_label_amount.keys():
			for Ci in x_label_amount[i].keys():
				for j in x_label_amount[i][Ci]:
					#print(x_label_amount[i][Ci][j])
					#print(x_dattolab_dict[i][Ci])
					x_label_amount[i][Ci][j] /= x_dattolab_dict[i][Ci]
		#print(x_label_amount)
		self.x_amount    = x_amount
		self.x_label_amount = x_label_amount
		
	def map_pred(self, feature):
		def pri(Ci, i):
			lamda = 0.000000000001
			j = feature[i]
			#print(j)
			a = self.x_label_amount[i][Ci][j]
			#print(a)
			b = self.x_amount[i][j]+lamda

			#print('%.5f'  %  (a*b))
			return a / b 
		
		a = list(map(lambda Ci:
					np.sum(list(map(lambda i: pri(Ci, i), range(0, len(feature))))),
					self.C_list))
		
		#print(a)
		
		return list(map(lambda Ci:
					np.sum(list(map(lambda i: pri(Ci, i), range(0, len(feature))))),
					self.C_list))
		
	def predict_p(self, X):
		preds_p = []
		for feature in X:
			#print(feature)
			pred_p = self.map_pred(feature = feature)
			#print(pred_p)
			preds_p.append(pred_p)
		#print(preds_p)
		return np.array(preds_p)

	def predict(self, X):
		preds = []
		for pripredict in self.predict_p(X):
			#print(pripredict)
			index  = np.argmax(pripredict)
			pred = self.C_list[index]
			preds.append(pred)
		#print(preds)
		return preds

def main():
	
	X,y = load_wine(return_X_y=True)
	trn = Map_classifier(X,y)
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state = 40)
	trn.learn(X_train,y_train)
	y_pred = trn.predict(X_test)
	accuracy=metrics.accuracy_score(y_test,y_pred)
	print('The MAP prediction result is {} %'.format(accuracy*100))
	
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(X, y) 
	print('The KNN prediction result is {} %'.format(neigh.score(X_train,y_train)*100))
	
	a =[]
	a_label = []
	with open('Xdataset.txt','r') as f:
		for line in f:
			a.extend([line.split() for line in f])
	for i in range (19):
		a_label.append(1)
	for k in range (19):
		a_label.append(2)

	with open('Ydataset.txt','r') as k:
		for line in k:
			a.extend([line.split() for line in k])
	
	X_train1,X_test1,y_train1,y_test1 = train_test_split(a,a_label,test_size = 0.8,random_state = 40)
	trn1 = Map_classifier(a,a_label)
	trn1.learn(X_train1,y_train1)
	y_pred1 = trn1.predict(X_test1)
	accuracy1=metrics.accuracy_score(y_test1,y_pred1)
	print('The GAUSS prediction result is {} %'.format(accuracy1*100))
	
main()