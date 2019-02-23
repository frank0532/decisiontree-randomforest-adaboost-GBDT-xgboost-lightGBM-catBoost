# Illustrate decision tree and its derivatives

## 1. Decision tree, code in "decisionTree.py"

### 1.1 What is decision tree? Build an "if-elif-else" condition tree according to some certain data which contains useful information that is really used to predict for future new data;
>> ![](https://github.com/frank0532/decision_tree_and_its_derivatives/blob/master/figs/data2decision_tree.png)

### 1.2 How to build it, i.e. how to arange features(columns' names of "Data") on the tree, some features at first layers and others later? 

When coming to the decision tree in Fig_1, why is "outlook" on the top rather than "Humidity" or "Wind"? In the fact the top one is selected by certain algorithms, such as ID3, C4.5, Cart and so on.

#### 1.2.1 ID3
>> ![](https://github.com/frank0532/decision_tree_and_its_derivatives/blob/master/figs/ID3.png)
#### 1.2.2 C4.5
>> ![](https://github.com/frank0532/decision_tree_and_its_derivatives/blob/master/figs/C4.5.png)
#### 1.2.3 Cart
>> ![](https://github.com/frank0532/decision_tree_and_its_derivatives/blob/master/figs/Cart.png)

## 2. Random Forest, code in "randomForest.py"

Select n_r lines and n_c columns randomly from training data matrix to re-build a new data matrix; doing like this n times, n new data matrixes can be there, that is D1~Dn;
>> ![](https://github.com/frank0532/decision_tree_and_its_derivatives/blob/master/figs/randomForest.png)

Each new data matrix "Di" (and its related label) can be used to train a decision tree, and at last n decision trees are there; these decision trees are grouped into a random forest; Now this random forest can be used to predict a new sample which would be predicted by each decision tree in the random forest independently; and the final predict of random forest is decision trees' majority vote result, i.e. decision trees' predicts are [1,1,1,-1], then the final predict is '1'; and if [-1,1,-1,-1], the final predict is '-1';

## 3. Adaboost, code in "adaboost.py"

As random forest, adaboost also has many decision trees but difference is that these decision trees are dependent one by one in order because of the dependent sample weights for different decision trees. Take for example, an adaboost is formed by just 3 decision trees as below.

>> ![](https://github.com/frank0532/decision_tree_and_its_derivatives/blob/master/figs/adaboost.png)

Training(mainly used for classifier, label is '1' or '-1'): the first decision tree is trained just as common decision tree because weights of all samples are the same (that is w1), and then get α1（details would be explained later）; the second decision tree is trained with sample weights:w2 which is updated by w1, label, predict1(the fisrt decision tree's predict) and α1, and then also get α2; the third decision tree is trained like the second one, and also get α3.

Predicting: the first decision tree predicts a new sample as predict1, the second decision tree predicts it as predict2 and also predict3; then if α1*predict1+α2*predict2+α3*predict3 >0, the label for this new sample should be '1',  or it should be '-1'.

Then what is α and how to get it?
>>![](https://github.com/frank0532/decision_tree_and_its_derivatives/blob/master/figs/alpha.png)

At last why α and w should be like that?
>>![](https://github.com/frank0532/decision_tree_and_its_derivatives/blob/master/figs/deduce-fx.png)
>>![](https://github.com/frank0532/decision_tree_and_its_derivatives/blob/master/figs/deduce-alpha-w.png)

## 4. GBDT, code in "GBDT.py" and "GBDT_LR.py"

As both random forest and adaboost, GBDT also has many decision trees but the special is that the labelS for each tree are different and different labels for these decision trees are dependent in order. Take for example, GBDT is formed by just 3 decision trees as below.
>>![](https://github.com/frank0532/decision_tree_and_its_derivatives/blob/master/figs/GBDT.png)










