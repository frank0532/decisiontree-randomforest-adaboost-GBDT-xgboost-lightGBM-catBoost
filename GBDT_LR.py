# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd
import numpy as np


data=pd.read_csv('../data/housing.csv')
data=data.sample(frac=1.0,replace=False)
splitPoint=len(data)//5
Xtest,Xtrain,ytest,ytrain=data.iloc[:splitPoint,:-1].values,data.iloc[splitPoint:,:-1].values,\
    data.iloc[:splitPoint:,-1].values,data.iloc[splitPoint:,-1].values
lgb_train=lgb.Dataset(Xtrain,ytrain)
lgb_eval=lgb.Dataset(Xtest,ytest)
params={
#        'boosting_type':'gbdt',
        'objective':'regression',
        'max_depth':7,
        'max_bin':200,
        'num_leaves':20,
        'learning_rate':0.03,
        'seed':100        
        }
num_leaf=20

rounds=[180,200,230,260,300,330,360]
errorList=[]
for roundi in rounds:
    gbm=lgb.train(params,lgb_train,num_boost_round=roundi,valid_sets=lgb_eval)
    gbm.save_model('gbmModel.txt')
    ypred=gbm.predict(Xtrain,pred_leaf=True)
    train_matrix=np.zeros([len(ypred),len(ypred[0])*num_leaf],dtype=np.int64)
    for i in range(0,len(ypred)):
        tmp=np.arange(len(ypred[0]))*num_leaf+np.array(ypred[i])
        train_matrix[i][tmp]+=1
    
    ypred=gbm.predict(Xtest,pred_leaf=True)
    test_matrix=np.zeros([len(ypred),len(ypred[0])*num_leaf],dtype=np.int64)
    for i in range(0,len(ypred)):
        tmp=np.arange(len(ypred[0]))*num_leaf+np.array(ypred[i])
        test_matrix[i][tmp]+=1
    
    lm=LinearRegression()
    lm.fit(train_matrix,ytrain)
    ytest_pred=lm.predict(test_matrix)
    errorList.append(((ytest_pred-ytest)**2).mean())

plt.figure(figsize=(15,8))
plt.plot(errorList)
plt.xlabel('tree numbers')
plt.ylabel('error')
ax=plt.gca()
ax.set_xticks(range(len(rounds)))
ax.set_xticklabels(rounds)





















