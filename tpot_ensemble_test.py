import numpy as np 
import pandas as pd 
import random
import config
import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

#from sklearn.utils import shuffle
#from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import StandardScaler
import featuretools as ft 
import featuretools.variable_types as vtypes 
from featuretools.primitives import make_agg_primitive, make_trans_primitive
from featuretools.variable_types import Numeric

from bayes_opt import BayesianOptimization

from mlens.ensemble import SuperLearner

from tpot import TPOTClassifier

import xgboost as xgb
#from boruta import BorutaPy
from boostaroota import BoostARoota

# can we axe the warnings?
def warn(*args, **kwargs):
	pass
import warnings
from sklearn.utils.testing import ignore_warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

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

# yay! A function to help us select features based on correlation :)
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

# a new function for us to eliminate correlated models
# just returns a list of the indices of the models we want to *keep*
def model_correlation(feature_matrix, correlation_threshold=0.95):
    """Feature selection for a dataframe."""

    print('Removing columns by correlation:')
    
    n_features_start = feature_matrix.shape[1]
    print(' Original shape: ', feature_matrix.shape)

    # then make a list of the range for the features
    keep_feature_idx_list = range(n_features_start)
    
    # Zero variance
    unique_counts = pd.DataFrame(feature_matrix.nunique()).sort_values(0, ascending = True)
    zero_variance_cols = list(unique_counts[unique_counts[0] == 1].index)
    n_zero_variance_cols = len(zero_variance_cols)

    # Remove zero variance columns
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in zero_variance_cols]]
    print(' {} zero variance columns.'.format(n_zero_variance_cols))
    
    # Correlations
    corr_matrix = feature_matrix.corr()

    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

    # Select the features with correlations above the threshold
    # Need to use the absolute value
    to_drop_ind = [int(column) for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

    n_collinear = len(to_drop_ind)
    
    #feature_matrix = feature_matrix[[x for x in feature_matrix if x not in to_drop]]
    print(' {} collinear columns removed with threshold: {}.'.format(n_collinear,
                                                                          correlation_threshold))
    
    total_removed = n_zero_variance_cols + n_collinear
    
    print(' Total columns removed: ', total_removed)
    #print('Shape after feature selection: {}.'.format(feature_matrix.shape))

    to_keep_ind = list(set(keep_feature_idx_list) - set(to_drop_ind))

    return to_keep_ind

### hiii we'll need a function that can take in a list of models/pipelines
### fit them on the dataset stratified cv folds
### and then assemble the OOF predictions
### perform model correlation on those
### and return the corrected OOF predictions as well as the corrected model list
def train_pred_model_list(layer_list, X, y, test_set):

	skf = StratifiedKFold(n_splits=config.config['num_folds'], shuffle=True)

	# create a zeroed array for all the preds to go in
	overall_preds = np.zeros((X.shape[0], len(layer_list)))

	overall_preds_test = np.zeros((test_set.shape[0], len(layer_list)))

	overall_preds.flags.writeable = True

	print('Training {} folds and gathering predictions:'.format(config.config['num_folds']))

	splits = list(skf.split(X, y))

	c = 0

	for model in layer_list:

		fold_count = 0

		print(' Training model {}'.format(c + 1))
		#print(' {}'.format(model))

		
		#split_num = 0

		# loop through all the indices we have
		for train_idxs, test_idxs in splits:

			#split_num += 1

			#print('  Split {}'.format(split_num))

			#start = time.time()
			model.fit(X[train_idxs], y[train_idxs])

			# adding onto this because I think predict_proba gives a 2D array?
			try:
				preds = model.predict_proba(X[test_idxs])[:, 1]
			except AttributeError:
				preds = model.predict(X[test_idxs])
			

			# add these to the np array
			# doesn't look like we can slice easily for this
			for count_i, ii in np.ndenumerate(test_idxs):

				overall_preds[ii, c] = preds[count_i[0]]

		c += 1
	
	# then go through the models again and just predict on the test set
	c = 0
	print(' Transforming the test set')
	for model in layer_list:

		try:
			preds_test = model.predict_proba(test_set)[:, 1]
		except AttributeError:
			preds_test = model.predict(test_set)

		overall_preds_test[:, c] = preds_test
		c += 1

	# cool now we should have a populated overall_preds
	# let's cut this down
	overall_preds = np.nan_to_num(overall_preds)
	overall_preds_df = pd.DataFrame(overall_preds)
	#hidden_pred_df[str(i)] = hidden_list[i].predict(X_test)

	if(config.config['correlation_model_elimination']):
		print(' Calculating the model correlation')
		to_keep_ind = model_correlation(overall_preds_df, correlation_threshold=0.95)

		layer_list = [layer_list[i] for i in to_keep_ind]

		# and then cut down the preds too
		final_preds = np.take(overall_preds, to_keep_ind, axis=1)
		final_test = np.take(overall_preds_test, to_keep_ind, axis=1)

		return layer_list, final_preds, final_test

	else:
		return layer_list, overall_preds, overall_preds_test

# function to read the csv and extract a list of only the feature names
def fetch_feature_list():

	# open the file
	f = open('feature_search.csv', 'r')

	# list to store the features. Start it with some of the better base features
	feature_list = ['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8']

	# go through the file
	for line in f:

		linelist = line.split(',')

		feat_name = linelist[0]

		if feat_name != '':
			feature_list.append(feat_name)

	f.close()
	return feature_list

# finally let's import the data
df = pd.read_csv("creditcard.csv")
df = df.drop(['Time'], axis=1) #,'V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
df = df.dropna()

# before we get into things, let's do all the featuretools definitions
def abs_log(column):
	return np.log(np.abs(column) + 1)
al = make_trans_primitive(function=abs_log, input_types=[Numeric], return_type=Numeric)

def squared(column):
	return np.square(column)
sq = make_trans_primitive(function=squared, input_types=[Numeric], return_type=Numeric)

def bins_5(column):
	temp = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform').fit([column])
	return temp.transform([column])
b5 = make_trans_primitive(function=bins_5, input_types=[Numeric], return_type=Numeric)

def binarize(column):
	temp = preprocessing.KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform').fit([column])
	return temp.transform([column])
bnz = make_trans_primitive(function=binarize, input_types=[Numeric], return_type=Numeric)

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

##### here's a list of all the trans_primitives
'''
['add_numeric', 'cum_mean', 'not_equal', 'haversine', 
	'cum_sum', 'equal', 'less_than_scalar', 'less_than_equal_to', 
	'multiply_boolean', 'greater_than_equal_to_scalar', 
	'multiply_numeric', 'diff', 'greater_than_scalar', 
	'modulo_numeric_scalar', 'subtract_numeric_scalar', 
	'divide_numeric_scalar', 'add_numeric_scalar', 'divide_by_feature', 
	'subtract_numeric', 'cum_min', 'not_equal_scalar', 'cum_count', 
	'equal_scalar', 'divide_numeric', 'less_than_equal_to_scalar', 
	'percentile', 'greater_than', 'less_than', 'multiply_numeric_scalar', 
	'greater_than_equal_to', 'modulo_by_feature', 'scalar_subtract_numeric_feature', 
	'isin', 'absolute', 'modulo_numeric'],
'''

feature_matrix, feature_names = ft.dfs(entityset=es, target_entity='obs',
										#agg_primitives = ['min', 'max', 'mean', 'count', 'sum', 'std', 'trend'],
										trans_primitives = ['add_numeric', 'less_than_scalar', 'less_than_equal_to', 
															'greater_than_equal_to_scalar', 'multiply_numeric', 
															'greater_than_scalar', 'subtract_numeric_scalar', 
															'divide_numeric_scalar', 'add_numeric_scalar', 
															'divide_by_feature', 'subtract_numeric', 'divide_numeric', 
															'less_than_equal_to_scalar', 'percentile', 'greater_than', 
															'less_than', 'multiply_numeric_scalar', 'greater_than_equal_to', 
															'modulo_by_feature', 'scalar_subtract_numeric_feature', 'absolute', 
															'modulo_numeric'],
										max_depth=1,
										n_jobs=1,
										verbose=1)

# alright here is where we're going to want to cut down all the variables
feature_list = fetch_feature_list()

# now filter out only the feature list
X = feature_matrix[feature_list]
del feature_matrix
X = X.fillna(X.mean())
y = df.pop('Class')
del df

'''
# eliminate features if they're too correlated before we get into boruta
if config.config['correlation_feature_elimination']:
	feature_matrix = feature_selection(feature_matrix, correlation_threshold = 0.7)
	print()
	print('Columns after feature engineering and correlation elimination:')
	print(list(feature_matrix.columns))

df_ = feature_matrix # make a copy of this
df_ = df_.dropna(how='any', axis=1)
y = df.pop('Class')

X = df_ # and another copy. Might not need this

print()
print('Starting Boruta')

br = BoostARoota(metric='logloss')
br.fit(X, y)
chosen_features = br.keep_vars_

##### So if we want to output feature importances
##### It looks like we'll need to train an xgb model again
##### and output the feature importances from that
ind = range(len(br.keep_vars_))
xgb_for_feat_imp = xgb.train(dtrain = xgb.DMatrix(X[chosen_features], label=y), params={})
ft_imps = pd.DataFrame.from_dict(xgb_for_feat_imp.get_score(importance_type='gain'), orient='index', columns=['importance']).sort_values('importance', ascending=True)

print()
print('Chosen features and importances:')
print(ft_imps)

### here is where we need to put these features and importances into a file
### you should be able to just output the dataframe to a csv?
### try that out and see what happens
ft_imps.to_csv('test_csv.csv')
'''

# split these out
# this isn't going to be shuffled because that's a mess.
#X_train, X_test, y_train, y_test = train_test_split(X[chosen_features].to_numpy(), y.to_numpy(), test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2)

# now let's make a bunch of lists of tpots
# hold all the base trained pipelines
base_list = []

# list of lists to hold the hidden pipelines for each layer
hidden_lol = []

# let's choose some of these numbers that are going to be assigned randomly
num_hidden_layers = random.randint(config.config['num_hidden_layers'][0], config.config['num_hidden_layers'][1])
num_base = random.randint(config.config['num_base'][0], config.config['num_base'][1])

print()
print('Training {} base TPOT pipelines'.format(num_base))

base_pred_df = pd.DataFrame()
for i in range(num_base):

    # we need to make a dummy dataset
    rand_weight_list = [random.random(), random.random()]
    x_dummy, y_dummy = make_classification(n_features = len(feature_list), n_informative = random.randint(2, 5), weights = rand_weight_list)
    
    base_list.append(TPOTClassifier(generations=config.config['base_num_gens'], 
    								population_size=config.config['base_pop_size'], 
    								scoring=config.config['metric'], 
    								cv=config.config['base_cv'], 
    								n_jobs=-1,
    								config_dict=config.base_models,
    								verbosity=0).fit(x_dummy, y_dummy).fitted_pipeline_)

base_list, base_preds, base_test = train_pred_model_list(base_list, X_train, y_train, X_test)

# keep track of this
# need the length of the base list since it's been updated
prev_num_hidden = len(base_list)

# go into a loop for this one!
for layer_num in range(num_hidden_layers):

	# just the list of pipelines for this layer
	hidden_list = []

	# and we need to choose the number we're going to have for this layer
	num_hidden = random.randint(config.config['num_hidden'][0], config.config['num_hidden'][1])

	print()
	print('Training {} hidden TPOT pipelines'.format(num_hidden))
	for i in range(num_hidden):

	    rand_weight_list = [random.random(), random.random()]

	    # need to use the number of models from the previous layer
	    x_dummy, y_dummy = make_classification(n_features = prev_num_hidden, n_informative = random.randint(2, 5), weights = rand_weight_list)
	    
	    hidden_list.append(TPOTClassifier(generations=config.config['hidden_num_gens'], 
	    									population_size=config.config['hidden_pop_size'], 
	    									scoring=config.config['metric'], 
	    									cv=config.config['hidden_cv'], 
	    									n_jobs=-1,
	    									config_dict=config.hidden_models,
	    									verbosity=0).fit(x_dummy, y_dummy).fitted_pipeline_)

	if layer_num == 0:
		hidden_list, hidden_preds, hidden_test = train_pred_model_list(hidden_list, base_preds, y_train, base_test)
	else:
		hidden_list, hidden_preds, hidden_test = train_pred_model_list(hidden_list, hidden_preds, y_train, hidden_test)

	# then when we're all done we'll append this whole layer to the hidden_lol
	hidden_lol.append(hidden_list)

	# update
	prev_num_hidden = len(hidden_list)

ens = SuperLearner(verbose=2, folds=config.config['num_folds'])

ens.add(base_list, proba=True)

# for the hidden lists we'll need a loop
for hidden_list in hidden_lol:
	ens.add(hidden_list, proba=True)

voting_list = []

# so here's what we'll do
# cv split into 5 parts
# that's going to be hard-coded for now
# we're going to make each tpot voting pipeline on one of the folds
# and then we're going to throw away the rest of the folds.
# deal?

##### YOU SHOULD STILL OPTIMIZE TPOTS ON THE WHOLE SET OK
#skf = StratifiedKFold(n_splits=5, shuffle=True)
#splits = skf.split(hidden_preds, y_train)
#train_idxs, test_idxs = list(splits)[0]

##### YOU NEED TO CHANGE THIS SO THAT HIDDEN PREDS ISN'T ALWAYS USED TO FIT
##### CAN'T USE HIDDEN PREDS IF THERE'S NO HIDDEN LAYER
for c in range(config.config['num_voters']):

	print('Training Voter {}/{}'.format(c + 1, config.config['num_voters']))
	voting_list.append(TPOTClassifier(generations=config.config['voting_num_gens'], 
										population_size=config.config['voting_pop_size'], 
										cv=config.config['voting_cv'], 
										scoring=config.config['metric'], 
										n_jobs=-1,
										config_dict=config.voting_models,
										verbosity=2).fit(hidden_preds, y_train).fitted_pipeline_)
	print()


# let's try zipping the voting list with a string
str_index_list = [str(i) for i in range(len(voting_list))]

voters_zipped = list(zip(str_index_list, voting_list))

# we need to train the models on each fold
v_model = VotingClassifier(voters_zipped)

# a list to store the voting models in
voter_list = []

# redo this with new splits
skf = StratifiedKFold(n_splits=5, shuffle=True)
splits = list(skf.split(hidden_preds, y_train))

print()
print('Training voting model across 5 splits')
for train_idxs, test_idxs in splits:

	voter_list.append(v_model.fit(hidden_preds[train_idxs], y_train[train_idxs]))

# so the order of this list matches up with the order of the test_idxs!

print()
print('Optimizing weights for voting classifier')

# we need a function to optimize
def opt_func(**weight_dict):

	# list to store the weights we'll be using for voting
	weights = []

	# variable to sum the total errors
	total_error = 0

	for model_idx in weight_dict.keys():

		weights.append(weight_dict[model_idx])

	# now we should have the list of weights we're using
	# reassign these
	for i in range(len(voter_list)):
		
		# each iteration in here represents a whole list of voting models

		voter_list[i].weights = weights
		#v_model.weights = weights

		# pull out the test idxs
		train_idxs, test_idxs = splits[i]

		# generate the predictions
		temp_preds = voter_list[i].predict(hidden_preds[test_idxs])

		# create the error from this
		# this uses just the hidden preds and y_train right?
		total_error += error(temp_preds, y_train[test_idxs])

	return total_error/len(voter_list)

# what are the pbounds going to be?
# just (0, 1) for each one of the voter weights
voter_pbounds = {}

for i in range(config.config['num_voters']):

	voter_pbounds['weight{}'.format(i)] = (0, 1)

optimizer = BayesianOptimization(
            f=opt_func,
            pbounds=voter_pbounds)
optimizer.maximize(init_points=5, n_iter=config.config['meta_learner_its'], xi=0.1)

max_params = optimizer.max
weight_dict = max_params['params']
final_weights = []

for model_idx in weight_dict.keys():

		final_weights.append(weight_dict[model_idx])

ens.add_meta(VotingClassifier(voters_zipped, weights=final_weights))

print()
print('Refitting the whole model with the new meta layer!')
ens.fit(X_train, y_train)
print()
print('Final predictions')

train_preds = ens.predict(X_train)
optim_preds = ens.predict(X_test)
#final_preds = ens.predict(X_holdout)

print()
print('Training score = {}'.format(error(train_preds, y_train)))
print('Test score = {}'.format(error(optim_preds, y_test)))
#print('Holdout score = {}'.format(error(final_preds, y_holdout)))

##### now we should save the model please
##### do several runs and save the best one
##### saving the features!
##### clustering as a feature???
##### randomizing the balance of the generated data for training initial pipelines
##### we shouldn't require at least 1 hidden layer
##### for training the voting models, what if we cv split the data coming in
#####  so we may have 5 sets of trained voters
#####  each set of trained voters has its associated test set
#####  each of those sets (weighted) predicts on their test set
#####  ah nuts how do you do TPOT for that?
#####  you could just use one fold to train the model and use that same model for the other folds
#####  you just have to retrain. on the other 4 folds
#####  So one of the models might have an advantage on its fold but the others don't necessarily
#####  This way you can eliminate one of the sets of data that you're train/test splitting
#####  and you get generate more robust scores for weight optimization


