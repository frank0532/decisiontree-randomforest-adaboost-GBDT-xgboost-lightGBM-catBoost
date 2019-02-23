# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_hastie_10_2
import matplotlib.pyplot as plt

class randomForest():
    def __init__(self,type_fit,n_estimators=30,max_features=5):
        self.type_fit=type_fit
        self.n_estimators=n_estimators
        self.max_features=max_features
        
        self.trees=[]
        self.featureix=[]
        for _ in range(n_estimators):
            if self.type_fit=='regression':
                self.trees.append(DecisionTreeRegressor())
            else:
                self.trees.append(DecisionTreeClassifier())
    def fit(self,X,y):
        rows,cols=X.shape
        sub_len=rows*4//5
        for i in range(self.n_estimators):
            ri=np.random.choice(rows,sub_len)
            ci=np.random.choice(cols,self.max_features,replace=False)
            self.trees[i].fit(X[ri][:,ci],y[ri])
            self.featureix.append(ci)
            
    def predict(self,X):
        preds=np.zeros([X.shape[0],self.n_estimators])
        for i,tree in enumerate(self.trees):
            preds[:,i]=tree.predict(X[:,self.featureix[i]])
        if self.type_fit=='regression':
            return preds.mean(axis=1)
        else:
            abs_min=abs(preds.min())
            preds+=abs_min
            return [np.bincount(pi.astype('int')).argmax()-abs_min for pi in preds]

if __name__=='__main__':
    #classifier
    X,y=make_hastie_10_2()
    Ldata=len(y)
    tmp=np.random.choice(Ldata,Ldata,replace=False)
    X=X[tmp]
    y=y[tmp]
    splitPoint=Ldata//5
    Xtest,Xtrain,ytest,ytrain=X[:splitPoint],X[splitPoint:],y[:splitPoint],y[splitPoint:]    
    precise_trains,precise_tests=[],[]
    roundsRange=range(1,100,10)
    for ni in roundsRange:
        rfi=randomForest('classifier',ni)
        rfi.fit(Xtrain,ytrain)
        pred_train=rfi.predict(Xtrain)
        precise_trains.append(sum((ytrain==pred_train))/len(pred_train))
        pred_test=rfi.predict(Xtest)
        precise_tests.append(sum((ytest==pred_test))/len(pred_test))
    results=pd.DataFrame([precise_trains,precise_tests],index=['Train','Test']).T
    fig=results.plot(figsize=(10,8),linewidth=3,color=['red','blue'],grid=True)
    fig.set_xlabel('tree numbers',fontsize=12)
    fig.set_xticks(range(len(roundsRange)))
    fig.set_xticklabels(roundsRange)
    fig.set_ylabel('precision',fontsize=12)
    fig.set_title('Classify Random Forest',fontsize=16)
    
#    #regression
#    data=pd.read_csv('E:/codes/housing.csv')
#    data=data.sample(frac=1.0,replace=False)
#    splitPoint=len(data)//5
#    Xtest,Xtrain,ytest,ytrain=data.iloc[:splitPoint,:-1].values,data.iloc[splitPoint:,:-1].values,\
#        data.iloc[:splitPoint:,-1].values,data.iloc[splitPoint:,-1].values
#    error_trains,error_tests=[],[]
#    roundsRange=range(1,50,5)
#    for ni in roundsRange:
#        rfi=randomForest('regression',ni)
#        rfi.fit(Xtrain,ytrain)
#        error_trains.append(np.mean((ytrain-rfi.predict(Xtrain))**2))
#        error_tests.append(np.mean((ytest-rfi.predict(Xtest))**2))
#    errors=pd.DataFrame([error_trains,error_tests],index=['train','test']).T
#    fig=errors.plot(figsize=(15,8),linewidth=3,color=['red','blue'],grid=True,)
#    fig.set_xlabel('tree numbers')
#    fig.set_xticks(range(len(roundsRange)))
#    fig.set_xticklabels(roundsRange)
#    fig.set_ylabel('Error')
#    fig.set_title('Regression Random Forest',fontsize=16)



    

        
    
        
        















































