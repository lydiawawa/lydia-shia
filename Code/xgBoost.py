#This model may not be used in analysis due to out of scope for ML One

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import xgboost as xgb

x_train, x_test, y_train, y_test = train_test_split(train_select,Target,test_size=0.30, random_state=30,stratify=Target)


x_train, x_test, y_train, y_test = train_test_split(train_select,Target,test_size=0.80,
                                                    random_state=30,stratify=Target)
dtrain1 = xgb.DMatrix(x_train,y_train,missing = -9999)
dtest1 = xgb.DMatrix(x_test,missing = -9999)
num_round = 125

#dtrain = xgb.DMatrix(train_select,Target,missing = -9999)
#dtest = xgb.DMatrix(test_select,missing = -9999)
num_round = 125

param1 = {'max_depth':7, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.75,'colsample_bytree':0.85}

param2 = {'max_depth':6, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.85,'colsample_bytree':0.75}

param3 = {'max_depth':8, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.65,'colsample_bytree':0.75}

param4 = {'max_depth':9, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.55,'colsample_bytree':0.65}

param5 = {'max_depth':12, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':1,'colsample_bytree':1}

bst1 = xgb.train(param1, dtrain, num_round)
bst2 = xgb.train(param2, dtrain, num_round)
bst3 = xgb.train(param3, dtrain, num_round)
bst4 = xgb.train(param3, dtrain, num_round)
bst5 = xgb.train(param3, dtrain, num_round)

ypred_submit = (bst1.predict(dtest) + bst2.predict(dtest) + bst3.predict(dtest) +  bst4.predict(dtest) +  bst5.predict(dtest))/5
print(log_loss(y_test, ypred_submit))

