import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import StandardScaler


import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


cwd = os.getcwd()
os.chdir("/Users/lydiawawa/Documents/Dat6202/Final/Final-Project-Group4/Code/Datasets/")
pets = pd.read_csv("trainK.csv")
pets = pd.DataFrame(pets)

pets.head()
pets.describe(include= "all")
# pets.drop ('OutcomeSubtype', axis=1, inplace=True)
pets.drop ('Aggressive', axis=1, inplace=True)
pets.drop ('At Vet', axis=1, inplace=True)
pets.drop ('Barn', axis=1, inplace=True)
pets.drop ('Behavior', axis=1, inplace=True)
pets.drop ('Court/Investigation', axis=1, inplace=True)
pets.drop ('Enroute', axis=1, inplace=True)
pets.drop ('Foster', axis=1, inplace=True)
pets.drop ('In Foster', axis=1, inplace=True)
pets.drop ('In Kennel', axis=1, inplace=True)
pets.drop ('In Surgery', axis=1, inplace=True)
pets.drop ('Medical', axis=1, inplace=True)
pets.drop ('Offsite', axis=1, inplace=True)
pets.drop ('Partner', axis=1, inplace=True)
pets.drop ('Rabies Risk', axis=1, inplace=True)
pets.drop ('SCRP', axis=1, inplace=True)
pets.drop ('Suffering', axis=1, inplace=True)
pets.drop ('AnimalID', axis=1, inplace=True)
pets.drop ('BreedName', axis=1, inplace=True)
pets.drop ('ageperiod', axis=1, inplace=True)


print(pets.isnull().sum(), '\n There are no missing')

Target = pets['Target']
outcome = pets['outcome']
pets.drop ('Target', axis=1, inplace=True)
pets.drop ('outcome', axis=1, inplace=True)
pets.drop ('color', axis=1, inplace=True)
# pets.groupby("outcome").count()
pets.info()
#
# pets_X = pets.iloc[:,:-1]
# pets_X = pets_X.drop('AnimalID', 1)
#
# ## Scale variables
# stdsc = StandardScaler()
#
# stdsc.fit(pets_X)
#
# X_std = stdsc.transform(pets_X)
#
# X = X_std

# Y = pets.values[:, -1]
# Y = Y.tolist()
X_train, X_test, y_train, y_test = train_test_split(pets, Target, test_size=0.3, random_state=100)

# X_train.to_csv('trainX',encoding='utf-8', index=False)
#%%-----------------------------------------------------------------------
# perform training
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
params = {}
clf =  MultinomialNB()
# glg = GaussianNB()

gs = GridSearchCV(clf, cv=k_fold, param_grid=params, return_train_score=True)
# creating the classifier object


# performing training
gs.fit(X_train, y_train)
gs.score(X_test,y_test)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)

gs.param_grid = {'alpha': [0.1, 0.01, 1, 2]}

gs.fit(X_train, y_train)
gs.score(X_test,y_test)

#%%-----------------------------------------------------------------------
# make predictions
random.seed( 10 )
# predicton on test
y_pred = gs.predict(X_test)

y_pred_score = gs.predict_proba(X_test)

# #%%--By Target(2 levels---------------------------------------------------------------------
from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(y_test, y_pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

print("\n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")


print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")


y_preds = y_pred_score[:,1]
y_preds = y_preds.tolist()
print("ROC_AUC : ", roc_auc_score(y_test,y_preds) * 100)
print("\n")

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = Target.unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()

# #%%--By Outcome(5 levels---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(pets, outcome, test_size=0.3, random_state=100)

# X_train.to_csv('trainX',encoding='utf-8', index=False)
#%%-----------------------------------------------------------------------
# perform training
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
params = {}
clf =  MultinomialNB()
# glg = GaussianNB()

gs = GridSearchCV(clf, cv=k_fold, param_grid=params, return_train_score=True)
# creating the classifier object


# performing training
gs.fit(X_train, y_train)
gs.score(X_test,y_test)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)

gs.param_grid = {'alpha': [0.1, 0.01, 1, 2]}

gs.fit(X_train, y_train)
gs.score(X_test,y_test)

#%%-----------------------------------------------------------------------
# make predictions
random.seed( 10 )
# predicton on test
y_pred = gs.predict(X_test)

y_pred_score = gs.predict_proba(X_test)


print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

print("\n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")


print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

#
# y_preds = y_pred_score[:,1]
# y_preds = y_preds.tolist()
# print("ROC_AUC : ", roc_auc_score(y_test,y_preds) * 100)
# print("\n")

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = outcome.unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()


