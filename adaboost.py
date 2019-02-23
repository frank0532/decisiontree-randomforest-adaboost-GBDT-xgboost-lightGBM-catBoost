# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 12:48:20 2019

@author: 10155195
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_hastie_10_2
import matplotlib.pyplot as plt

class adaboost():
    def __init__(self,type_fit,n_estimators=100,max_depth=5):
        self.type_fit=type_fit
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.alphas=[]
        self.trees=[]

    def fit(self,X,y):
        Ltrain=len(y)
        w=np.ones(Ltrain)/Ltrain
        learnRatio=0.007
        for _ in range(self.n_estimators):
            if self.type_fit=='regression':
                cdf=np.cumsum(w)
                cdf/=cdf[-1]
                random_state=np.random.RandomState(0)
                uniform_samples=random_state.random_sample(X.shape[0])
                bootstrap_idx=cdf.searchsorted(uniform_samples,side='right')
                bootstrap_idx=np.array(bootstrap_idx,copy=False)
                rawTree=DecisionTreeRegressor(max_depth=self.max_depth)
                rawTree.fit(X[bootstrap_idx],y[bootstrap_idx],sample_weight=w[bootstrap_idx])
                self.trees.append(rawTree)
                predi=rawTree.predict(X)   
                error_vect=np.abs(predi-y)
                error_max=error_vect.max()
                error_vect/=error_max
#                if self.loss=='square':
#                    error_vect**=2
#                elif self.loss=='exponential':
#                error_vect**=2
                error_vect=1.-np.exp(-error_vect)
                err=(w*error_vect).sum()          
                beta=err/(1.-err)
                alpha=learnRatio*np.log(1./beta)
                self.alphas.append(alpha)
                w*=np.power(beta,(1.-error_vect)*learnRatio)
                w=w/w.sum()

            else:
                rawTree=DecisionTreeClassifier(max_depth=self.max_depth)
                rawTree.fit(X,y,sample_weight=w)
                self.trees.append(rawTree)
                predi=rawTree.predict(X)
                misMatch=(predi!=y)*1.0
                err=sum(w*misMatch)
                err=np.clip(err,1e-15,1-1e-15)
                alphai=0.5*np.log((1-err)/err)
                self.alphas.append(alphai)
                w=w*np.exp([alphai if i>0 else -alphai for i in misMatch])
                w/=sum(w)

    def predict(self,X):
        Ltest=len(X)
        preds=np.zeros(Ltest)
        if self.type_fit=='regression':
            if len(self.alphas)<2:
                return self.trees[0].predict(X)
            else:
                preds=[]
                for treei in self.trees:
                    preds.append(treei.predict(X))
                preds=np.array(preds).T
                sorted_idx=np.argsort(preds,axis=1)
                weight_cdf=np.cumsum(np.array(self.alphas)[sorted_idx],axis=1)
                median_or_above=(weight_cdf>=0.5)*weight_cdf[:,-1][:,np.newaxis]
                median_idx=median_or_above.argmax(axis=1)                
                median_estimators=sorted_idx[np.arange(X.shape[0]),median_idx]
                return preds[np.arange(X.shape[0]), median_estimators]
        else:
            for i in range(self.n_estimators):
                predi=self.trees[i].predict(X)
                preds=[preds[i2]+predi[i2]*self.alphas[i] for i2 in range(Ltest)]
            return np.sign(preds)

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
    roundsRange=range(1,300,20)
    for ni in roundsRange:
        adbi=adaboost('classifier',ni,2)
        adbi.fit(Xtrain,ytrain)
        pred_train=adbi.predict(Xtrain)
        precise_trains.append(sum((ytrain==pred_train))/len(pred_train))
        pred_test=adbi.predict(Xtest)
        precise_tests.append(sum((ytest==pred_test))/len(pred_test))
    results=pd.DataFrame([precise_trains,precise_tests],index=['Train','Test']).T
    fig=results.plot(figsize=(10,8),linewidth=3,color=['red','blue'],grid=True)
    fig.set_xlabel('boost numbers',fontsize=12)
    fig.set_xticks(range(len(roundsRange)))
    fig.set_xticklabels(roundsRange)
    fig.set_ylabel('precision',fontsize=12)
    fig.set_title('adaboost training',fontsize=16)

#    #regression
#    data=pd.read_csv('E:/codes/housing.csv')
#    data=data.sample(frac=1.0,replace=False)
#    splitPoint=len(data)//5
#    Xtest,Xtrain,ytest,ytrain=data.iloc[:splitPoint,:-1].values,data.iloc[splitPoint:,:-1].values,\
#        data.iloc[:splitPoint:,-1].values,data.iloc[splitPoint:,-1].values
#    error_trains,error_tests=[],[]
#    roundsRange=range(1,5000,200)
#    for ni in roundsRange:
#        adbi=adaboost('regression',ni,3)
#        adbi.fit(Xtrain,ytrain)
#        error_trains.append(np.mean((ytrain-adbi.predict(Xtrain))**2))
#        error_tests.append(np.mean((ytest-adbi.predict(Xtest))**2))
#    errors=pd.DataFrame([error_trains,error_tests],index=['train','test']).T
#    fig=errors.plot(figsize=(15,8),linewidth=3,color=['red','blue'],grid=True,)
#    fig.set_xlabel('boost numbers')
#    fig.set_xticks(range(len(roundsRange)))
#    fig.set_xticklabels(roundsRange)
#    fig.set_ylabel('Error')
#    fig.set_title('Adaboost',fontsize=16)

    









    
    
    
    
    






