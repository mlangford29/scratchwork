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
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

from joblib import dump

import featuretools as ft 
import featuretools.variable_types as vtypes 
from featuretools.primitives import make_agg_primitive, make_trans_primitive
from featuretools.variable_types import Numeric


from mlens.ensemble import SuperLearner

from tpot import TPOTClassifier

import xgboost as xgb
#from boruta import BorutaPy
from boostaroota import BoostARoota


base_models = {

    # Classifiers
    'sklearn.naive_bayes.GaussianNB': {
    },

    'sklearn.dummy.DummyClassifier': {
        'strategy': ['stratified', 'prior']
    },

    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': np.arange(1e-3, 1.001, 1e-3),
        'fit_prior': [True, False]
    },

    'sklearn.naive_bayes.ComplementNB': {
        'alpha': np.arange(1e-3, 1.001, 1e-3),
        'fit_prior': [True, False],
        'norm': [True, False]
    },

    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': np.arange(1e-3, 1.001, 1e-3),
        'fit_prior': [True, False]
    },

    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': range(2, 25),
        'criterion': ["gini", "entropy"],
        #'max_features': np.arange(0.2, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': range(2, 25),
        'criterion': ["gini", "entropy"],
        #'max_features': np.arange(0.2, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': range(2,10),
        'learning_rate': np.arange(1e-3, 1.001, 1e-3),
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        #'max_features': np.arange(0.2, 1.01, 0.05)
    },

    'sklearn.ensemble.AdaBoostClassifier': {
        'n_estimators': range(2,10),
        'learning_rate': np.arange(1e-3, 1.001, 1e-3),
    },    

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

    'sklearn.svm.SVC': {
        'tol': np.arange(1e-5, 1e-1, 1e-4),
        'C': np.arange(1e-3, 1.001, 1e-3),
        'max_iter': range(1, 1000),
        'probability': [True]
    },

    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': np.arange(1e-5, 1, 1e-4),
        'dual': [True, False]
    },

    'sklearn.linear_model.LogisticRegressionCV': {
        'penalty': ["l1", "l2"],
        'Cs': np.arange(1e-5, 10, 1e-4),
    },

    'xgboost.XGBClassifier': {
        'n_estimators': range(2,10),
        'max_depth': range(1, 11),
        'learning_rate': np.arange(1e-3, 1.001, 1e-3),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'nthread': [1]
    },

    'sklearn.neural_network.MLPClassifier': {
        'hidden_layer_sizes':[(10,), (20,), (30,), (40,), (50,), (60,), (70,), (80,), (90,), (100,), 
                            (10,5), (20,10), (30,15), (40,20), (50,25), (60,30), (70,35), (80,40), (90,45), (100,50)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.0, 0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': range(10, 100)
    },

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

    # 'sklearn.preprocessing.PolynomialFeatures': {
    #     'degree': [2],
    #     'include_bias': [False],
    #     'interaction_only': [False]
    # },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    'tpot.builtins.ZeroCount': {
    },

    # 'tpot.builtins.OneHotEncoder': {
    #     'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
    #     'sparse': [False],
    #     'threshold': [10]
    # },

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
        'step': np.arange(0.05, .5, 0.05),
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
                'n_estimators': range(2,100),
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

highest_score = 0
winning_model = None

# new idea
# what if we just specify the number of layers 
# and weight for number of models in each layer
num_layers = 3
layer_weights = [5, 3, 2]

# they should probably all have the same meta model. Let's do LGBM
from lightgbm import LGBMClassifier as lgbm

for its in range(5):
    
    pipe_opt = TPOTClassifier(generations=5, 
                            population_size=5, 
                            cv=2, 
                            scoring='f1', 
                            n_jobs=-1,
                            config_dict=base_models,
                            verbosity=1)

    x_dummy, y_dummy = make_classification(n_features = len(X_train[0]))
    pipe_opt = pipe_opt.fit(x_dummy, y_dummy)
    eval_ind_not = pipe_opt.eval_ind.copy()
    temp = [pipe_opt._toolbox.compile(expr=eval_ind_not[i]) for i in range(len(eval_ind_not))]
    

    # trying out 10 different random configs
    for i in range(10):
        
        ens = SuperLearner(verbose=1)

        eval_ind = temp.copy()
        random.shuffle(eval_ind)
        num_to_slot = len(pipe_opt.eval_ind)

        # now we need to find the num in each layer
        weights_total = sum(layer_weights)

        for j in range(len(layer_weights)):

            if j == len(layer_weights) - 1:

                # this means we're at the last layer! Put everything else in
                ens.add(eval_ind)
                continue

            num_in_layer = int(layer_weights[j]/weights_total*num_to_slot)

            layerlist = []

            for k in range(num_in_layer):
                layerlist.append(eval_ind.pop())

            ens.add(layerlist)

        # then add the meta model
        ens.add_meta(lgbm(n_estimators=1000, verbose=-1, learning_rate=.005))

        try: 
            ens.fit(X_train, y_train)
            train_score = f1_score(ens.predict(X_train), y_train)
            test_score = f1_score(ens.predict(X_test), y_test)
            real_score = train_score * test_score
            print(' Training score is {}'.format(train_score))
            print(' Testing score is {}'.format(test_score))
            print(' Real score is {}'.format(real_score))
        except:
            print(' There was an error with this one. Throwing it out')
            continue

        if real_score > highest_score:
            print(' New highest score found!')
            highest_score = real_score
            winning_model = ens


'''
for its in range(5):
    pipe_opt = TPOTClassifier(generations=10, 
            				population_size=10, 
            				cv=2, 
            				scoring='f1', 
            				n_jobs=-1,
            				config_dict=base_models,
            				verbosity=1)

    x_dummy, y_dummy = make_classification(n_features = len(X_train[0]))
    pipe_opt = pipe_opt.fit(x_dummy, y_dummy)

    # copy the list so we can remove from it later
    eval_ind_not = pipe_opt.eval_ind.copy()
    temp = [pipe_opt._toolbox.compile(expr=eval_ind_not[i]) for i in range(len(eval_ind_not))]
    eval_ind = []

    # pre-train all of the models and kick out the ones that don't work
    print(' pre-training')
    for model in temp:
        try:
            model.fit(X_train, y_train)
            eval_ind.append(model)
        except:
            print(' encountered a problem with a model. Throwing it out') 
            continue

    eval_ind_holder = eval_ind.copy()
    print(' done pre-training')

    for i in range(10):
        eval_ind = eval_ind_holder.copy()
        random.shuffle(eval_ind)
        num_to_slot = len(pipe_opt.eval_ind)

        ens = SuperLearner(verbose=1)

        while num_to_slot > 0:

            layerlist = []

            # choose a random number for this layer
            if num_to_slot < 5:
                for _ in range(len(eval_ind)):
                    layerlist.append(eval_ind.pop())
                num_to_slot = 0

            else:

                num_in_layer = random.randint(5, num_to_slot)

                for _ in range(num_in_layer):
                    layerlist.append(eval_ind.pop())

                num_to_slot -= num_in_layer

            if(num_to_slot > 0):
                ens.add(layerlist)
            else:
                str_index_list = [str(i) for i in range(len(layerlist))]
                voters_zipped = list(zip(str_index_list, layerlist))
                ens.add_meta(VotingClassifier(voters_zipped))

        try: 
            ens.fit(X_train, y_train)
            train_score = f1_score(ens.predict(X_train), y_train)
            test_score = f1_score(ens.predict(X_test), y_test)
            real_score = train_score * test_score
            print(' Training score is {}'.format(train_score))
            print(' Testing score is {}'.format(test_score))
            print(' Real score is {}'.format(real_score))
        except:
            print(' There was an error with this one. Throwing it out')
            continue

        if real_score > highest_score:
            print(' New highest score found!')
            highest_score = real_score
            winning_model = ens
'''

real_temp = int(highest_score * 100)
dump(winning_model, 'testedpipes_{}_f1_traintest.joblib'.format(highest_score))

print()
print('Best model found')
print(ens)
print('Highest score: {}'.format(highest_score))




