# -*- coding: utf-8 -*-

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
                pass
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
            pass
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


    









    
    
    
    
    






