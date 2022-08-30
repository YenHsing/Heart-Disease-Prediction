from sklearn.datasets import load_wine
from sklearn import metrics
from sklearn.model_selection import train_test_split

import map_classifier


# Load Iris data set
X, y = load_wine(return_X_y=True)

# Create a classifier
clf = map_classifier.MAPClassifier()

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=40)

# Learn then model
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Calc accuracy
test_accuracy = metrics.accuracy_score(y_test, y_pred)

# Print the accuracy
print(test_accuracy)

a =[]
a_label = []

#with open('winedata.txt','r') as f:
#    for line in f:
#        a.extend([line.split(',') for line in f])
#    print(a[2])
#    
#for i in range(177):
#    del(a[i][0])
#print(len(a))


#for i in range (59):
#    a_label.append(0)
#for k in range (71):
#    a_label.append(1)
#for k in range (47):
#    a_label.append(2)

#clf1 = map_classifier.MAPClassifier()
#X_train1,X_test1,y_train1,y_test1 = train_test_split(a,a_label,test_size = 0.5,random_state = 9487)
#clf1.fit(X_train1, y_train1)
#y_pred1 = clf1.predict(X_test1)
#test_accuracy1 = metrics.accuracy_score(y_test1, y_pred1)
#print(test_accuracy1)
    