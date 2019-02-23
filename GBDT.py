# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_hastie_10_2
from sklearn import linear_model
import matplotlib.pyplot as plt

class GBDT():
    def __init__(self,type_fit='regression',learning_rate=0.1,n_estimators=100,max_depth=5):
        self.type_fit=type_fit
        self.learning_rate=learning_rate
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.trees=[]
        self.alphas=[]
        for _ in range(n_estimators):
            self.trees.append(DecisionTreeRegressor(max_depth=self.max_depth))
    def fit(self,X,y):
        lineM=linear_model.LinearRegression()
        y_pred=np.zeros(len(y))
        for i in range(self.n_estimators):
            if self.type_fit=='regression':
                gradient=y_pred-y
            else:
                y_pred=np.clip(y_pred,1e-15,1.5)
                gradient=(1 - y)/(1 - y_pred)- (y / y_pred)
                
            self.trees[i].fit(X,gradient)
            update=self.trees[i].predict(X)
            lineM.fit(update[:,np.newaxis],y-y_pred)
            alpha=lineM.coef_[0]
            self.alphas.append(alpha)
            y_pred+=self.learning_rate*alpha*update

            
    def predict(self,X):
        y_pred=np.zeros(len(X))
        for i in range(self.n_estimators):
            y_pred+=self.alphas[i]*self.learning_rate*self.trees[i].predict(X)
        if self.type_fit=='regression':
            return y_pred
#            return np.around(y_pred)
        else:
            return np.around(y_pred)

if __name__=='__main__':
    #classifier
    X,y=make_hastie_10_2()
    Ldata=len(y)
    tmp=np.random.choice(Ldata,Ldata,replace=False)
    X=X[tmp]
    y=y[tmp]
    y=np.array([1 if yi>0 else 0 for yi in y])
#    y=pd.get_dummies(y).values
    splitPoint=Ldata//5
    Xtest,Xtrain,ytest,ytrain=X[:splitPoint],X[splitPoint:],y[:splitPoint],y[splitPoint:]    
    precise_trains,precise_tests=[],[]
    roundsRange=range(1,100,10)
    for ni in roundsRange:
        gbdti=GBDT('classifier',0.1,ni)
        gbdti.fit(Xtrain,ytrain)
        pred_train=gbdti.predict(Xtrain)
        precise_trains.append(sum((ytrain==pred_train))/len(pred_train))
        pred_test=gbdti.predict(Xtest)
        precise_tests.append(sum((ytest==pred_test))/len(pred_test))
    results=pd.DataFrame([precise_trains,precise_tests],index=['Train','Test']).T
    fig=results.plot(figsize=(10,8),linewidth=3,color=['red','blue'],grid=True)
    fig.set_xlabel('boost numbers',fontsize=12)
    fig.set_xticks(range(len(roundsRange)))
    fig.set_xticklabels(roundsRange)
    fig.set_ylabel('precision',fontsize=12)
    fig.set_title('GBDT training',fontsize=16)

#    #regression
#    data=pd.read_csv('E:/codes/housing.csv')
#    data=data.sample(frac=1.0,replace=False)
#    splitPoint=len(data)//5
#    Xtest,Xtrain,ytest,ytrain=data.iloc[:splitPoint,:-1].values,data.iloc[splitPoint:,:-1].values,\
#        data.iloc[:splitPoint:,-1].values,data.iloc[splitPoint:,-1].values
#    error_trains,error_tests=[],[]
#    roundsRange=range(1,150,20)
#    for ni in roundsRange:
#        gbdti=GBDT('regression',0.02,ni)
#        gbdti.fit(Xtrain,ytrain)
#        error_trains.append(np.mean((ytrain-gbdti.predict(Xtrain))**2))
#        error_tests.append(np.mean((ytest-gbdti.predict(Xtest))**2))
#    errors=pd.DataFrame([error_trains,error_tests],index=['train','test']).T
#    fig=errors.plot(figsize=(15,8),linewidth=3,color=['red','blue'],grid=True,)
#    fig.set_xlabel('boost numbers')
#    fig.set_xticks(range(len(roundsRange)))
#    fig.set_xticklabels(roundsRange)
#    fig.set_ylabel('Error')
#    fig.set_title('GBDT',fontsize=16)
    
    
    
    
    
    
    
    
    
