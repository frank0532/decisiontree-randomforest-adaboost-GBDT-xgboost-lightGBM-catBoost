# merge_randomForest_adaboost_GBDT
Merge algorithms of random forest, adaboost and DBDT together into one by two methods:

1. Simplified:

i.	Create “random forest” which includes n_1 “big Trees”;

ii.	Each “big Tree” in “random forest” is created by n_2 trees which are combined by “GBDT”;

iii.	Each tree in “GBDT” is trained by “sample_weight” according to “Adaboost”;

Note: this method seems to run well until now;

2. Completed: 

i.	Create “random forest” which includes n_1 “big Trees”;

ii.	Each “big Tree” in “random forest” is created by n_2 “big Trees” which are combined by “Adaboost”;

iii.	Each “big Tree” in “Adaboost” is created by n_3 trees which are combined by “GBDT”;

iv.	Each tree in “GBDT” is trained according to “GBDT” completely;

Note: this method seems to have some problems in “Adaboost” stage.
