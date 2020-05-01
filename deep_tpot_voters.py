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

from joblib import dump

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

config = {

	# number of voters. For now we'll have this as just an int
	'num_voters':10,

	# number of bayesian opt iterations we'll optimize voting weights for
	'meta_learner_its':100,

	# what metric will we be using to evaluate?
	'metric':'f1',

	# voting TPOT parameters
	'voting_num_gens':10,
	'voting_pop_size':10,
	'voting_cv':5,

}

config_dict = {
	# Classifiers
    'sklearn.naive_bayes.GaussianNB': {
    },

    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': np.arange(1e-3, 1.001, 1e-3),
        'fit_prior': [True, False]
    },

    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': np.arange(1e-3, 1.001, 1e-3),
        'fit_prior': [True, False]
    },

    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 50),
        'min_samples_split': range(1, 100),
        'min_samples_leaf': range(1, 100)
    },


 #    'sklearn.ensemble.ExtraTreesClassifier': {
 #        'n_estimators': range(5, 500),
 #        'criterion': ["gini", "entropy"],
 #        #'max_features': np.arange(0.2, 1.01, 0.05),
 #        'min_samples_split': range(1, 100),
 #        'min_samples_leaf': range(1, 100),
 #        'bootstrap': [True, False]
 #    },

 #    'sklearn.ensemble.RandomForestClassifier': {
 #        'n_estimators': range(2, 500),
 #        'criterion': ["gini", "entropy"],
 #        #'max_features': np.arange(0.2, 1.01, 0.05),
 #        'min_samples_split': range(1, 100),
 #        'min_samples_leaf':  range(1, 100),
 #        'bootstrap': [True, False]
 #    },

 #    'sklearn.ensemble.GradientBoostingClassifier': {
 #        'n_estimators': range(2, 500),
 #        'learning_rate': np.arange(1e-3, 1.001, 1e-3),
 #        'max_depth': range(1, 50),
 #        'min_samples_split': range(1, 100),
 #        'min_samples_leaf': range(1, 100),
 #        'subsample': np.arange(0.05, 1.01, 0.05),
 #        #'max_features': np.arange(0.2, 1.01, 0.05)
 #    },

	# 'sklearn.ensemble.AdaBoostClassifier': {
 #        'n_estimators': range(2, 500),
 #        'learning_rate': np.arange(1e-3, 1.001, 1e-3),
 #    },    

 #    'sklearn.svm.LinearSVC': {
 #        'penalty': ["l1", "l2"],
 #        'loss': ["hinge", "squared_hinge"],
 #        'dual': [True, False],
 #        'tol': np.arange(1e-5, 1e-1, 1e-4),
 #        'C': np.arange(1e-3, 1.001, 1e-3),
 #        'max_iter': range(10, 50000)
 #    },

 #    'sklearn.svm.SVC': {
 #        'tol': np.arange(1e-5, 1e-1, 1e-4),
 #        'C': np.arange(1e-3, 1.001, 1e-3),
 #        'max_iter': range(10, 50000),
 #        #'probability': [True]
 #    },

 #    'sklearn.linear_model.LogisticRegression': {
 #        'penalty': ["l1", "l2"],
 #        'C': np.arange(1e-5, 1, 1e-4),
 #        'dual': [True, False]
 #    },

 #    'xgboost.XGBClassifier': {
 #        'n_estimators': range(2, 500),
 #        'max_depth': range(1, 50),
 #        'learning_rate': np.arange(1e-3, 1.001, 1e-3),
 #        'subsample': np.arange(0.05, 1.01, 0.05),
 #        'min_child_weight': range(1, 100),
 #        'nthread': [1]
 #    },

    'lightgbm.LGBMClassifier': {
    	'boosting_type': ['gbdt', 'dart', 'goss', 'rf'],
    	'num_leaves': range(1, 50),
    	'max_depth': range(1, 50),
    	'learning_rate': np.arange(1e-3, 1.001, 1e-3),
    	'n_estimators': range(2, 500),
    	'subsample_for_bin': range(1000, 500000),
    	'min_child_samples': range(1, 100),
    	'subsample': np.arange(0.05, 1.01, 0.05),
    	'reg_alpha': np.arange(0, 0.99, 1e-3),
    	'reg_lambda': np.arange(0, 0.99, 1e-3),
    	'verbose': [-1]
    },

    'catboost.CatBoostClassifier': {
    	'iterations': range(1, 10000),
    	'learning_rate': np.arange(1e-3, 1.001, 1e-3),
    	'reg_lambda': np.arange(0, 0.99, 1e-3),
    	'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS', 'Poisson', 'No'],
    	'bagging_temperature': np.arange(0, 10, 1e-3),
    	'use_best_model': [True, False],
    	'best_model_min_trees': range(1, 500),
    	'depth': range(1, 50),
    	'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
    	'max_leaves': range(1, 50),
    	'task_type': ['GPU']
    },

    # 'sklearn.linear_model.SGDClassifier': {
    #     'loss': ['log', 'hinge', 'modified_huber', 'squared_hinge', 'perceptron'],
    #     'penalty': ['elasticnet'],
    #     'alpha': np.arange(1e-3, 1.001, 1e-3),
    #     'learning_rate': ['invscaling', 'constant'],
    #     'fit_intercept': [True, False],
    #     'l1_ratio': np.arange(1e-3, 1, 1e-3),
    #     'eta0': np.arange(1e-3, 1.001, 1e-3),
    #     'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    # },

    # 'sklearn.neural_network.MLPClassifier': {
    #     'hidden_layer_sizes':[(100,), (200,), (300,), (400,), (500,), (600,), (700,), (800,), (900,), (1000,), 
    #     					(100,50), (200,100), (300,150), (400,200), (500,250), (600,300), (700,350), (800,400), (900,450), (1000,500)],
    #     'activation': ['identity', 'logistic', 'tanh', 'relu'],
    #     'solver': ['lbfgs', 'sgd', 'adam'],
    #     'alpha': np.arange(1e-3, 1.001, 1e-3),
    #     'learning_rate': ['constant', 'invscaling', 'adaptive'],
    #     'max_iter': range(1, 10000)
    # },

    # Preprocesssors
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.1, 1.01, 0.05)
    },

    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },

    'sklearn.preprocessing.MaxAbsScaler': {
    },

    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    'tpot.builtins.ZeroCount': {
    },

    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(20, 100),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

     'sklearn.feature_selection.VarianceThreshold': {
         'threshold': np.arange(1e-4, .05, 1e-4)
     },

    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': range(2,100),
                'criterion': ['gini', 'entropy'],
                #'max_features': np.arange(0.2, 1.01, 0.05)
            }
        }
    },

    'sklearn.feature_selection.SelectFromModel': {
        #'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': range(2,1000),
                'criterion': ['gini', 'entropy'],
                #'max_features': np.arange(0.2, 1.01, 0.05)
            }
        }
    }
}

# can we axe the warnings?
def warn(*args, **kwargs):
	pass
import warnings
from sklearn.utils.testing import ignore_warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# generic error function
def error(preds, y_test):
    
    error_name = config['metric']
    
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



# ok and then we'll do all the featuretools things that need to happen
es = ft.EntitySet(id = 'card') # no clue what this means but whatever

# make an entity from the observations data
es = es.entity_from_dataframe(dataframe = df.drop('Class', axis=1),
								entity_id = 'obs',
								index = 'index')

feature_matrix, feature_names = ft.dfs(entityset=es, target_entity='obs',
										#agg_primitives = ['min', 'max', 'mean', 'count', 'sum', 'std', 'trend'],
										trans_primitives = ['divide_by_feature', 'add_numeric', 'less_than_equal_to', 'greater_than_equal_to_scalar', 'multiply_numeric', 'subtract_numeric_scalar', 'divide_numeric_scalar', 'add_numeric_scalar', 'subtract_numeric', 'divide_numeric', 'percentile', 'greater_than', 'less_than', 'multiply_numeric_scalar', 'greater_than_equal_to', 'modulo_by_feature', 'scalar_subtract_numeric_feature', 'absolute', 'modulo_numeric'],
										max_depth=1,
										n_jobs=1,
										verbose=1)

# alright here is where we're going to want to cut down all the variables
feature_list = fetch_feature_list()

# now filter out only the feature list
X = feature_matrix[feature_list]
del feature_matrix
X = X.fillna(X.mean())
X = X*1.0 # convert all to float hopefully
y = df.pop('Class')
del df

# split these out
# this isn't going to be shuffled because that's a mess.
#X_train, X_test, y_train, y_test = train_test_split(X[chosen_features].to_numpy(), y.to_numpy(), test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2)

voting_list = []

for c in range(config['num_voters']):

	print('Training Voter {}/{}'.format(c + 1, config['num_voters']))
	voting_list.append(TPOTClassifier(generations=config['voting_num_gens'], 
										population_size=config['voting_pop_size'], 
										cv=config['voting_cv'], 
										scoring=config['metric'], 
										n_jobs=-1,
										config_dict=config_dict,
										verbosity=2).fit(X_train, y_train).fitted_pipeline_)
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
splits = list(skf.split(X_train, y_train))

print()
print('Training voting model across 5 splits')
for train_idxs, test_idxs in splits:

	voter_list.append(v_model.fit(X_train[train_idxs], y_train[train_idxs]))

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
		temp_preds = voter_list[i].predict(X_train[test_idxs])

		# create the error from this
		total_error += error(temp_preds, y_train[test_idxs])

	return total_error/len(voter_list)

# what are the pbounds going to be?
# just (0, 1) for each one of the voter weights
voter_pbounds = {}

for i in range(config['num_voters']):

	voter_pbounds['weight{}'.format(i)] = (0, 1)

optimizer = BayesianOptimization(
            f=opt_func,
            pbounds=voter_pbounds)
optimizer.maximize(init_points=5, n_iter=config['meta_learner_its'], xi=0.1)

max_params = optimizer.max
weight_dict = max_params['params']
final_weights = []

for model_idx in weight_dict.keys():

		final_weights.append(weight_dict[model_idx])

final_model = VotingClassifier(voters_zipped, weights=final_weights)

print()
print('Training final model on full training set')
final_model.fit(X_train, y_train)

train_preds = final_model.predict(X_train)
optim_preds = final_model.predict(X_test)

print()
print('Training score = {}'.format(error(train_preds, y_train)))
print('Test score = {}'.format(error(optim_preds, y_test)))

print()
print('Saving model')

test_score = error(optim_preds, y_test)
test_score = int(test_score * 100)

dump(final_model, '{}_voters_{}_f1.joblib'.format(config['num_voters'], test_score))


