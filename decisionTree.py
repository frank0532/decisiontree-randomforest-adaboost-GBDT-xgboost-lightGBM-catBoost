# -*- coding: utf-8 -*-

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydot

class decision_tree():
    def __init__(self,treeType='cart'):
        self.treeType=treeType
        self.Tree=None
        if self.treeType=='ID3':
            self.type_split=self.ID3_split
        elif self.treeType=='C4dot5':
            self.type_split=self.C4dot5_split
        elif self.treeType=='cart':
            self.type_split=self.cart_split
        else:
            self.type_split=self.regression_split
         
    #cart
    def cart_split(self,data):#need modify
        bestGini=100000.0
        bestFeature=0
        bestValue=0
        for featurei in data.columns[:-1]:
            featureList=data[[featurei,'quality']].set_index(featurei)
#            groupF=featureList.groupby(featurei)['quality']
#            p1=groupF.mean()
            featuresUnique=np.unique(featureList.index)
            for i2 in featuresUnique:
                p1=featureList.loc[i2]
                p2=featureList[featureList.index!=i2]
                gini1=1-np.mean(p1)**2-(1-np.mean(p1))**2
                gini2=1-np.mean(p2)**2-(1-np.mean(p2))**2
                c1=len(p1)
                c2=len(p2)
                Gi=c1/(c1+c2)*gini1+c2/(c1+c2)*gini2
                if Gi['quality']<bestGini:
                    bestGini=Gi['quality']
                    bestFeature=featurei
                    bestValue=i2
           
            
#            Glist=p1.map(lambda x:1-x**2-(1-x)**2)
#            cS=groupF.count()
#            rS=cS/cS.sum()
#            gini=(Glist*rS).sum()
#            if gini<bestGini:
#                bestGini=gini
#                bestFeature=featurei
        return (bestFeature,bestValue)
    #ID3
    def ID3_split(self,data):
        mostEntropy=0.0
        bestFeature=0
        tmp=data['quality'].mean()
        Entropy=-tmp*np.log2(tmp+1e-10)-(1-tmp)*np.log2(1-tmp+1e-10)
        for featurei in data.columns[:-1]:
            featureList=data[[featurei,'quality']]
            groupF=featureList.groupby(featurei)['quality']
            p1=groupF.mean()
            Ei=p1.map(lambda x: -x*np.log2(x+1e-10)-(1-x)*np.log2(1-x+1e-10))
            c1=groupF.count()
            Wi=c1/c1.sum()
            entropyi=(Ei*Wi).sum()
            EntropyAdd=Entropy-entropyi
            if EntropyAdd>mostEntropy:
                mostEntropy=EntropyAdd
                bestFeature=featurei
        return bestFeature
    #C4.5
    def C4dot5_split(self,data):
        mostEntropyRatio=0.0
        bestFeature=0
        tmp=data['quality'].mean()
        Entropy=-tmp*np.log2(tmp+1e-10)-(1-tmp)*np.log2(1-tmp+1e-10)
        for featurei in data.columns[:-1]:
            featureList=data[[featurei,'quality']]
            groupF=featureList.groupby(featurei)['quality']
            p1=groupF.mean()
            Ei=p1.map(lambda x: -x*np.log2(x+1e-10)-(1-x)*np.log2(1-x+1e-10))
            c1=groupF.count()
            Wi=c1/c1.sum()
            entropyi=(Ei*Wi).sum()
            EntropyAdd=Entropy-entropyi
            tmp=Wi.map(lambda x:-x*np.log2(x+1e-10)).sum()
            EntropyAddRatio=EntropyAdd/tmp
            if EntropyAddRatio>mostEntropyRatio:
                mostEntropyRatio=EntropyAddRatio
                bestFeature=featurei
        return bestFeature
    #regression
    def regression_split(self,data):
        minVar=10000.0
        bestFeature=0
        for col in data.columns[:-1]:
            tmp=(data[[col,'quality']].set_index(col)-pd.DataFrame(data.groupby(col)['quality'].mean())).var()['quality']
            if tmp<minVar:
                minVar=tmp
                bestFeature=col
        return bestFeature
    
    def create_tree(self,data):
        if len(data['quality'].unique())==1:
            return data['quality'].iloc[0]
        if data.shape[1]==1:
            if self.treeType=='regression':
                return data['quality'].mean()
            else:
                return round(data['quality'].mean())
        featureBest=self.type_split(data)        
        if self.treeType=='cart':
            valueBest=featureBest[1]
            featureBest=featureBest[0]            
            Tree={featureBest:{}}
            dataLeft=data[data[featureBest]==valueBest]
            dataRight=data[data[featureBest]!=valueBest]
            if len(dataLeft)<1:
                Tree[featureBest][valueBest]=round(data['quality'].mean())
            else:
                del dataLeft[featureBest]
                Tree[featureBest][valueBest]=self.create_tree(dataLeft)
            if len(dataRight)<1:
                Tree[featureBest]['not_'+valueBest]=round(data['quality'].mean())
            else:
                if len(dataRight[featureBest].unique())<2:
                    del dataRight[featureBest]
                Tree[featureBest]['not_'+valueBest]=self.create_tree(dataRight)                
        else:
            Tree={featureBest:{}}
            values=data[featureBest].unique()
            for valuei in values:
                subData=data[data[featureBest]==valuei]
                if len(subData)<1:
                    Tree[featureBest][valuei]=round(data['quality'].mean())
                else:
                    del subData[featureBest]
                    Tree[featureBest][valuei]=self.create_tree(subData)
        return Tree
    
    def fit(self,data):
        self.Tree=self.create_tree(data)

    def predict(self,test):
        predicts=[]
        for i in range(len(test)):
            ti=test.iloc[i]
            Tree=self.Tree
            while 1:
                keyi=list(Tree.keys())[0]
                if ti[keyi] in Tree[keyi].keys():
                    Tree=Tree[keyi][ti[keyi]]
                else:
                    Tree=Tree[keyi][list(Tree[keyi].keys())[1]]                        
                if type(Tree)!=dict:
                    if Tree==0:
                        predicts.append('bad')
                    elif Tree==1:
                        predicts.append('good')
                    else:
                        predicts.append(Tree)
                    break
                   
        return predicts
    def show_tree(self):
        def draw(parentName,childName):
            edge=pydot.Edge(parentName, childName)
            graph.add_edge(edge)
        def visit(node,parent=None):
            for k,v in node.items():
                k=str(k)
                if isinstance(v,dict):
                    if parent:
                        draw(parent,k)
                    visit(v,k)
                else:
                    draw(parent,k)
                    if v==0:
                        draw(k,k+'_'+'bad')
                    elif v==1:
                        draw(k,k+'_'+'good')
                    else:
                        draw(k,k+'_'+str(v))
        graph = pydot.Dot(graph_type='graph')
        visit(self.Tree)
        graph.write_png('temp.png')
        img=Image.open('temp.png')
        plt.figure(figsize=(25,18))
        plt.title(self.treeType+'_'+'tree_figure')
        plt.imshow(img)
        

if __name__=='__main__':
# classifier tree
    data=pd.read_excel('E:/codes/watermelon.xlsx')
    data['quality']=[1 if x=='good' else 0 for x in data['quality']]    
    train=data.loc[1:16]
    test=data.loc[[0,16]]
    DT=decision_tree()
## regression tree
#    data=pd.read_excel('E:/codes/watermelonRegression.xlsx')
#    train=data.loc[1:15]
#    test=data.loc[[0,16]]
#    DT=decision_tree('regression')
    
    DT.fit(train)
    Tree=DT.Tree
    print(DT.predict(test.iloc[:,:-1]))
    DT.show_tree()

















