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
from tpot import TPOTClassifier
import config

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# finally let's import the data
df = pd.read_csv("creditcard.csv")
df = df.drop(['Time'], axis=1) #,'V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
df = df.dropna()


def feature_selection(feature_matrix, missing_threshold=90, correlation_threshold=0.95):
    """Feature selection for a dataframe."""
    
    feature_matrix = pd.get_dummies(feature_matrix)
    n_features_start = feature_matrix.shape[1]
    print('Original shape: ', feature_matrix.shape)

    # Find missing and percentage
    missing = pd.DataFrame(feature_matrix.isnull().sum())
    missing['percent'] = 100 * (missing[0] / feature_matrix.shape[0])
    missing.sort_values('percent', ascending = False, inplace = True)

    # Missing above threshold
    missing_cols = list(missing[missing['percent'] > missing_threshold].index)
    n_missing_cols = len(missing_cols)

    # Remove missing columns
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in missing_cols]]
    print('{} missing columns with threshold: {}.'.format(n_missing_cols,
                                                                        missing_threshold))
    
    # Zero variance
    unique_counts = pd.DataFrame(feature_matrix.nunique()).sort_values(0, ascending = True)
    zero_variance_cols = list(unique_counts[unique_counts[0] == 1].index)
    n_zero_variance_cols = len(zero_variance_cols)

    # Remove zero variance columns
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in zero_variance_cols]]
    print('{} zero variance columns.'.format(n_zero_variance_cols))
    
    # Correlations
    corr_matrix = feature_matrix.corr()

    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

    # Select the features with correlations above the threshold
    # Need to use the absolute value
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

    n_collinear = len(to_drop)
    
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in to_drop]]
    print('{} collinear columns removed with threshold: {}.'.format(n_collinear,
                                                                          correlation_threshold))
    
    total_removed = n_missing_cols + n_zero_variance_cols + n_collinear
    
    print('Total columns removed: ', total_removed)
    print('Shape after feature selection: {}.'.format(feature_matrix.shape))
    return feature_matrix

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
										trans_primitives = ['divide_numeric_scalar', 'multiply_numeric'],
										max_depth=1,
										n_jobs=1,
										verbose=1)



feature_matrix = feature_selection(feature_matrix, correlation_threshold = 0.8)

df_ = feature_matrix # make a copy of this
df_ = df_.dropna(how='any', axis=1)
y = df.pop('Class')

X = df_ # and another copy. Might not need this

br = BoostARoota(metric='logloss')
br.fit(X, y)

print()
print('keep vars')
print(list(br.keep_vars_))
chosen_features = br.keep_vars_

print()
print(' Final chosen features:')
print(' {}'.format(chosen_features))

# split these out
# this isn't going to be shuffled because that's a mess.
X_temp, X_holdout, y_temp, y_holdout = train_test_split(X[chosen_features].to_numpy(), y.to_numpy(), test_size=0.10)
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.25)


best_pipe = TPOTClassifier(generations=50, 
    								population_size=10, 
    								scoring='f1', 
    								cv=3, 
    								n_jobs=-1,
    								#config_dict=config.base_models,
    								verbosity=1).fit(X_train, y_train).fitted_pipeline_

# generic error function
def error(preds, y_test):
    
    error_name = config.config['metric']
    
    if error_name == 'roc_auc':
        return roc_auc_score(preds, y_test)
    elif error_name == 'accuracy':
        return accuracy_score(preds, y_test)
    elif error_name == 'recall':
        return recall_score(preds, y_test)
    elif error_name == 'precision':
        return precision_score(preds, y_test)
    elif error_name == 'f1':
        return f1_score(preds, y_test)
    else:
        print('unsure what your metric is in the config so using accuracy instead')
        return accuracy_score(preds, y_test)



train_preds = best_pipe.predict(X_train)
optim_preds = best_pipe.predict(X_test)
final_preds = best_pipe.predict(X_holdout)

print()
print('Training score = {}'.format(error(train_preds, y_train)))
print('Optimizing score = {}'.format(error(optim_preds, y_test)))
print('Overall score = {}'.format(error(final_preds, y_holdout)))
