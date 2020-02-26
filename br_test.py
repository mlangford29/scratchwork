from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import pandas as pd 
import featuretools as ft 
import featuretools.variable_types as vtypes 
from featuretools.primitives import make_agg_primitive, make_trans_primitive
from featuretools.variable_types import Numeric
from boostaroota import BoostARoota
from sklearn.metrics import f1_score

# finally let's import the data
df = pd.read_csv("creditcard.csv")
df = df.drop(['Time'], axis=1) #,'V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
df = df.dropna()

# before we get into things, let's do all the featuretools definitions
def log_plus_one(column):
	return np.log(column + min(column) + 1)
lpo = make_trans_primitive(function=log_plus_one, input_types=[Numeric], return_type=Numeric)

def abs_log(column):
	return np.log(np.abs(column) + 1)
al = make_trans_primitive(function=abs_log, input_types=[Numeric], return_type=Numeric)

def squared(column):
	return np.square(column)
sq = make_trans_primitive(function=squared, input_types=[Numeric], return_type=Numeric)

def add_cols(numeric1, numeric2):
	return numeric1+numeric2
adc = make_trans_primitive(function=add_cols, input_types=[Numeric, Numeric], return_type=Numeric)

def add_abs_cols(numeric1, numeric2):
	return np.abs(numeric1) + np.abs(numeric2)
aac = make_trans_primitive(function=add_abs_cols, input_types=[Numeric, Numeric], return_type=Numeric)

def sqrt_square_sum(numeric1, numeric2):
	return np.sqrt(np.square(numeric1) + np.square(numeric2))
sss = make_trans_primitive(function=sqrt_square_sum, input_types=[Numeric, Numeric], return_type=Numeric)

### It would be interesting to see if having a third column in any of this changes things!

# ok and then we'll do all the featuretools things that need to happen
es = ft.EntitySet(id = 'card') # no clue what this means but whatever

# make an entity from the observations data
es = es.entity_from_dataframe(dataframe = df.drop('Class', axis=1),
								entity_id = 'obs',
								index = 'index')

feature_matrix, feature_names = ft.dfs(entityset=es, target_entity='obs',
										agg_primitives = ['min', 'max', 'mean', 'count', 'sum', 'std', 'trend'],
										trans_primitives = ['percentile', lpo, al, sq, adc, aac, sss],
										max_depth=1,
										n_jobs=1,
										verbose=1)




df_ = feature_matrix # make a copy of this
df_ = df_.dropna(how='any', axis=1)
y = df.pop('Class')

X = df_ # and another copy. Might not need this

br = BoostARoota(metric='logloss')
br.fit(X, y)

print()
print('keep vars')
print(br.keep_vars_)
