# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_hastie_10_2
from sklearn import linear_model
import matplotlib.pyplot as plt

class mergeModelSimplified():
    def __init__(self,n_random_forest=100,depth_random_forest=8,max_features=8,\
                 n_adaboost=100,n_GBDT=50,depth_GBDT=5,learning_ratio=0.1):
        self.n_random_forest=n_random_forest
        self.depth_random_forest=depth_random_forest
        self.max_features=max_features
        self.n_GBDT=n_GBDT
        self.depth_GBDT=depth_GBDT
        self.learning_ratio=learning_ratio
        self.cols_groups=[]
        self.tree_groups=[]
        self.alpha_groups=[]
        
    def fit(self,X,y):
        Ly=len(y)
        sub_len=Ly*4//5
        for i in range(self.n_random_forest):
            ri=np.random.choice(Ly,sub_len)
            ci=np.random.choice(X.shape[1],self.max_features,replace=False)
            X_sub,y_sub=X[ri][:,ci],y[ri]
            self.cols_groups.append(ci)
            trees=[]
            alphas=[]
            for i2 in range(self.n_GBDT):
                trees.append(DecisionTreeRegressor(max_depth=self.depth_GBDT))
            Ltrain=len(y_sub)
            y_pred=np.zeros(Ltrain)
            w=np.ones(Ltrain)/Ltrain
            lineM=linear_model.LinearRegression()
            for i2 in range(self.n_GBDT):
                y_pred=np.clip(y_pred,1e-15,1.5)
                gradients=(1-y_sub)/(1-y_pred)-y_sub/y_pred
                trees[i2].fit(X_sub,gradients,sample_weight=w)
                update=trees[i2].predict(X_sub)
                lineM.fit(update[:,np.newaxis],y_sub-y_pred)
                alphai2=lineM.coef_[0]
                alphas.append(alphai2)
                y_pred+=self.learning_ratio*alphai2*update
        
                err=np.dot(w,np.round(y_pred)!=y_sub)
                err=np.clip(err,1e-15,1-1e-15)
                alpha_adaboost=0.5*np.log((1.0-err)/err)
                w*=np.exp([alpha_adaboost if i3 else -alpha_adaboost for i3 in np.round(y_pred)!=y_sub])
                w/=sum(w)
#            print(np.mean(np.round(y_pred)!=y_sub))
            self.tree_groups.append(trees)
            self.alpha_groups.append(alphas)
            
            
    def predict(self,X):
        preds=np.zeros([X.shape[0],self.n_random_forest])
        for i,trees in enumerate(self.tree_groups):
            X_sub=X[:,self.cols_groups[i]]
            for i2,tree in enumerate(trees):
                preds[:,i]+=self.learning_ratio*self.alpha_groups[i][i2]*tree.predict(X_sub)
        preds=np.round(preds)
        return [np.bincount(pi.astype('int')).argmax() for pi in preds]

if __name__=='__main__':
    X,y=make_hastie_10_2()
    Ldata=len(y)
    y=np.array([1 if i>0 else 0 for i in y])
    tmp=np.random.choice(Ldata,Ldata,replace=False)
    X=X[tmp]
    y=y[tmp]
    splitPoint=Ldata//5
    Xtest,Xtrain,ytest,ytrain=X[:splitPoint],X[splitPoint:],y[:splitPoint],y[splitPoint:]    
    precise_trains,precise_tests=[],[]
    roundsRange=range(1,100,10)
#    roundsRange=[5]
    for ni in roundsRange:
        rfi=mergeModelSimplied(n_random_forest=ni,n_adaboost=ni,n_GBDT=ni//2+1)
        rfi.fit(Xtrain,ytrain)
        pred_train=rfi.predict(Xtrain)
        precise_trains.append(sum((ytrain==pred_train))/len(pred_train))
        pred_test=rfi.predict(Xtest)
        precise_tests.append(sum((ytest==pred_test))/len(pred_test))
    results=pd.DataFrame([precise_trains,precise_tests],index=['Train','Test']).T
    fig=results.plot(figsize=(10,8),linewidth=3,color=['red','blue'],grid=True)
    fig.set_xlabel('random forest numbers',fontsize=12)
    fig.set_xticks(range(len(roundsRange)))
    fig.set_xticklabels(roundsRange)
    fig.set_ylabel('precision',fontsize=12)
    fig.set_title('Classify by Merge Model Simplied',fontsize=16)
