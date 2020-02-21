

import numpy as np 
import pandas as pd 
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from bayes_opt import BayesianOptimization

import pandas as pd
import numpy as np 
import tensorflow as tf
#from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from mlens.ensemble import SuperLearner

from tpot import TPOTClassifier




df = pd.read_csv("creditcard.csv")

df = df.drop(['Time','V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
df = df.dropna()


train_df = df.sample(frac=0.8, random_state=0)
test_df =  df.drop(train_df.index)

train_df = shuffle(train_df)
test_df = shuffle(test_df)

y_train = train_df.pop('Class').to_numpy()
y_test = test_df.pop('Class').to_numpy()
X_train = train_df.to_numpy()
X_test = test_df.to_numpy()


# now let's make a bunch of lists of tpots
base_list = []
num_base = 5


##### Train each of the models before you get to the super learner
##### I think you want to export the model itself into a separate list or something
##### like the whole pipeline it finds


for i in range(num_base):
    
    base_list.append(TPOTClassifier(generations=2, population_size=2, scoring="f1", cv=2, n_jobs=-1, verbosity=1).fit(X_train[0:10000,:], y_train[0:10000]).fitted_pipeline_)


hidden_list = []
for i in range(3):
    
    hidden_list.append(TPOTClassifier(generations=2, population_size=2, scoring="f1", cv=2, n_jobs=-1, verbosity=1).fit(X_train[10000:20000,:], y_train[10000:20000]).fitted_pipeline_)

model = SuperLearner(verbose=2, folds=2)

print('adding the base list to the super learner')
model.add(base_list)
model.add(hidden_list)

print('adding meta learner')
model.add_meta(TPOTClassifier(generations=10, population_size=40, cv=5, scoring="f1", n_jobs=-1, verbosity=2))

print('fitting super learner!')
model.fit(X_train, y_train)
print('score = {}'.format(f1_score(model.predict(X_test), y_test)))
 

