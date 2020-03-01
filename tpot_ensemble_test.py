import numpy as np 
import pandas as pd 
import random
import config

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier


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

from boruta import BorutaPy
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
    
    n_features_start = feature_matrix.shape[1]
    print('Original shape: ', feature_matrix.shape)

    # then make a list of the range for the features
    keep_feature_idx_list = range(n_features_start)
    
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
    to_drop_ind = [int(column) for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

    n_collinear = len(to_drop)
    
    #feature_matrix = feature_matrix[[x for x in feature_matrix if x not in to_drop]]
    print('{} collinear columns removed with threshold: {}.'.format(n_collinear,
                                                                          correlation_threshold))
    
    total_removed = n_zero_variance_cols + n_collinear
    
    print('Total columns removed: ', total_removed)
    #print('Shape after feature selection: {}.'.format(feature_matrix.shape))

    to_keep_ind = list(set(keep_feature_idx_list) - set(to_drop_ind))

    return to_keep_ind


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

#feature_matrix = feature_matrix.dropna()

# eliminate features if they're too correlated before we get into boruta
if config.config['correlation_feature_elimination']:
	feature_matrix = feature_selection(feature_matrix, correlation_threshold = 0.5)
	print()
	print('Columns after feature engineering and correlation elimination:')
	print(list(feature_matrix.columns))

df_ = feature_matrix # make a copy of this
df_ = df_.dropna(how='any', axis=1)
y = df.pop('Class')

X = df_ # and another copy. Might not need this

print()
print('Starting Boruta')

# now let's do some boruta!!
'''
rfc = RandomForestClassifier(n_jobs = -1)
boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2, max_iter=config.config['max_iter_boruta'])
boruta_selector.fit(X.to_numpy(), y.to_numpy())

print()
print(' Number of selected features: {}'.format(boruta_selector.n_features_))

# now go through and actually select those features
total_cols = list(X.columns) # list of all the names
chosen = list(boruta_selector.support_) # this is array of booleans
chosen_features = [] # these are the NAMES of the ones we'll be using

for i in range(len(total_cols)):
	if chosen[i]:
		chosen_features.append(total_cols[i])
'''

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



##### OMG
##### RIGHT HERE WHILE YOU'RE TRAINING THEM
##### SAVE THE BEST PIPELINE ON THE SMALLER SET
##### TRAIN EACH OF THEM ON LARGER SET
##### AFTER ALL DONE, PREDICT
##### PUT IN PD DATAFRAME, CORRELATION ELIMINATION
##### ONLY SELECT THOSE THAT REMAIN
base_pred_df = pd.DataFrame()
for i in range(num_base):
    
    base_list.append(TPOTClassifier(generations=config.config['base_num_gens'], 
    								population_size=config.config['base_pop_size'], 
    								scoring=config.config['metric'], 
    								cv=config.config['base_cv'], 
    								n_jobs=-1,
    								config_dict=config.base_models,
    								verbosity=1).fit(X_train[0:5000,:], y_train[0:5000]).fitted_pipeline_)

    base_pred_df[str(i)] = base_list[i].predict(x_test)

to_keep_ind = model_correlation(base_pred_df, correlation_threshold=.98)
base_list = base_list[to_keep_ind]


# go into a loop for this one!
for _ in range(num_hidden_layers):

	# just the list of pipelines for this layer
	hidden_list = []

	# and we need to choose the number we're going to have for this layer
	num_hidden = random.randint(config.config['num_hidden'][0], config.config['num_hidden'][1])

	print()
	print('Training {} hidden TPOT pipelines'.format(num_hidden))
	hidden_pred_df = pd.DataFrame()
	for i in range(num_hidden):
	    
	    hidden_list.append(TPOTClassifier(generations=config.config['hidden_num_gens'], 
	    									population_size=config.config['hidden_pop_size'], 
	    									scoring=config.config['metric'], 
	    									cv=config.config['hidden_cv'], 
	    									n_jobs=-1,
	    									config_dict=config.hidden_models,
	    									verbosity=1).fit(X_train[5000:10000,:], y_train[5000:10000]).fitted_pipeline_)

	    hidden_pred_df[str(i)] = hidden_list[i].predict(x_test)

	to_keep_ind = model_correlation(hidden_pred_df, correlation_threshold=.98)
	hidden_list = hidden_list[to_keep_ind]

	# then when we're all done we'll append this whole layer to the hidden_lol
	hidden_lol.append(hidden_list)

ens = SuperLearner(verbose=2, folds=config.config['num_folds'])

ens.add(base_list)

# for the hidden lists we'll need a loop
for hidden_list in hidden_lol:
	ens.add(hidden_list)

# fit this and grab the predictions!
##### CONCERN THAT WE NEED TO DO THIS OVER MULTIPLE FOLDS. I'M NOT SURE IF THIS HAPPENS AUTOMATICALLY
print()
print('Training the base and hidden models to gather predictions!')
ens.fit(X_train, y_train)
hidden_preds = ens.predict(X_test)

voting_list = []

for _ in range(config.config['num_voters']):

	voting_list.append(TPOTClassifier(generations=config.config['voting_num_gens'], 
										population_size=config.config['voting_pop_size'], 
										cv=config.config['voting_cv'], 
										scoring=config.config['metric'], 
										n_jobs=-1,
										config_dict=config.voting_models,
										verbosity=2).fit(hidden_preds, y_test).fitted_pipeline_)

# let's try zipping the voting list with a string
str_index_list = [str(i) for i in range(len(voting_list))]

voters_zipped = list(zip(str_index_list, voting_list))

# now we need to optimize the weights
print()
print('Optimizing weights for voting classifier')

# maybe we can just pull out the preds once?
# then apply our own weighting
# like say we already have the preds and just wanna apply the weights

# oh excuse me we need to train the model
v_model = VotingClassifier(voters_zipped)

##### THIS IS CURRENTLY WRONG I JUST DON'T KNOW WHAT TO DO
##### WE NEED TO TRAIN ON THE CV-TEST SET FROM THE PREVIOUS LAYERS
v_model.fit(X_train, y_train)

# we need a function to optimize
def opt_func(**weight_dict):

	# list to store the weights we'll be using for voting
	weights = []

	for model_idx in weight_dict.keys():

		weights.append(weight_dict[model_idx])

	# now we should have the list of weights we're using
	# reassign these
	v_model.weights = weights

	temp_preds = v_model.predict(X_test)

	return error(temp_preds, y_test)

# what are the pbounds going to be?
# just (0, 1) for each one of the voter weights
voter_pbounds = {}

for i in range(config.config['num_voters']):

	voter_pbounds['weight{}'.format(i)] = (0, 1)

optimizer = BayesianOptimization(
            f=opt_func,
            pbounds=voter_pbounds)
optimizer.maximize(init_points=2, n_iter=config.config['meta_learner_its'], xi=0.5)

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
print('Final prediction')
final_preds = ens.predict(X_holdout)
print()
print('Overall score = {}'.format(error(final_preds, y_holdout)))

##### now we should save the model please







