# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_hastie_10_2
from sklearn import linear_model
import matplotlib.pyplot as plt

class mergeModel():
    def __init__(self,n_random_forest=100,depth_random_forest=8,max_features=8,\
                 n_adaboost=100,depth_adaboost=2,
                 n_GBDT=50,depth_GBDT=5,learning_ratio=0.1):
        self.n_random_forest=n_random_forest
        self.depth_random_forest=depth_random_forest
        self.max_features=max_features
        self.n_GBDT=n_GBDT
        self.depth_GBDT=depth_GBDT
        self.learning_ratio=learning_ratio
        self.n_adaboost=n_adaboost

        self.cols_group2=[]
        self.alpha_adaboost_group2=[]
        self.tree_group3=[]
        self.alpha_GBDT_group3=[]
        
    def fit(self,X,y):
        Ly=len(y)
        sub_len=Ly*4//5        
        for i in range(self.n_random_forest):
            ri=np.random.choice(Ly,sub_len)
            ci=np.random.choice(X.shape[1],self.max_features,replace=False)
            X_sub,y_sub=X[ri][:,ci],y[ri]
            self.cols_group2.append(ci)
            Ltrain=len(X_sub)
            w=np.ones(Ltrain)/Ltrain
            alpha_adaboost_group=[]
            tree_group2=[]
            alpha_GBDT_group2=[]
            for i2 in range(self.n_adaboost):
                y_pred=np.zeros(Ltrain)
                trees=[]
                alpha_GBDT=[]
                lineM=linear_model.LinearRegression()
                for i3 in range(self.n_GBDT):
                    y_pred=np.clip(y_pred,1e-15,1-1e-15)
                    gradients=(1-y_sub)/(1-y_pred)-y_sub/y_pred
                    tree=DecisionTreeRegressor(max_depth=self.depth_GBDT)
                    tree.fit(X_sub,gradients,sample_weight=w)
                    trees.append(tree)
                    update=tree.predict(X_sub)
                    lineM.fit(update[:,np.newaxis],y_sub-y_pred)
                    alphai3=lineM.coef_[0]
                    alpha_GBDT.append(alphai3)
                    y_pred+=self.learning_ratio*alphai3*update                
                miss_match=np.round(y_pred)!=y_sub
                print(np.mean(miss_match))
                
                err=np.dot(w,miss_match)
                err=np.clip(err,1e-15,1-1e-15)
                alphai2=0.5*np.log((1.0-err)/err)
                alpha_adaboost_group.append(alphai2)
                w*=np.exp([alphai2 if i3 else -alphai2 for i3 in miss_match])
                w/=sum(w)
                tree_group2.append(trees)
                alpha_GBDT_group2.append(alpha_GBDT)
            self.alpha_adaboost_group2.append(alpha_adaboost_group)
            self.tree_group3.append(tree_group2)
            self.alpha_GBDT_group3.append(alpha_GBDT_group2)           
            
    def predict(self,X):
        Ltrain=X.shape[0]
        preds=np.zeros([Ltrain,self.n_random_forest])
        for i in range(self.n_random_forest):
            ci=self.cols_group2[i]
            alpha_adaboost_group=self.alpha_adaboost_group2[i]
            tree_group2=self.tree_group3[i]
            alpha_GBDT_group2=self.alpha_GBDT_group3[i]
            predi=np.zeros(Ltrain)
            for i2 in range(self.n_adaboost):
                trees=tree_group2[i2]
                alpha_GBDT=alpha_GBDT_group2[i2]
                predi2=np.zeros(Ltrain)
                for i3 in range(self.n_GBDT):
                    predi2+=self.learning_ratio*trees[i3].predict(X[:,ci])*alpha_GBDT[i3]
#                predi2+=trees[i3+1].predict(X[:,ci])*alpha_GBDT[i3+1]
                predi+=np.round(predi2)*alpha_adaboost_group[i]
            preds[:,i]+=predi
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
    roundsRange=[10]
    for ni in roundsRange:
        rfi=mergeModel(n_random_forest=ni,n_adaboost=ni,n_GBDT=ni//2+1)
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
    fig.set_title('Classify by Merge Model',fontsize=16)