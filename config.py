# this is where you're going to put all of the config everythings!
import numpy as np

# a config for the run itself
config = {
	
	# number of 'hidden' layers. Probably not the right term
	# but these are the layers that are in between the base and the meta-learner
	'num_hidden_layers':(2, 2),

	# number of models in the base. This is a range
	'num_base':(20, 20),

	# number of models in hidden layers. This is a range
	'num_hidden':(20, 20),

	# number of voters. For now we'll have this as just an int
	'num_voters':3,

	# are we going to do feature elimination based on correlation?
	'correlation_feature_elimination':True,
	'correlation_model_elimination':False,

	# how many iterations for Boruta to run
	'max_iter_boruta':200,

	# number of bayesian opt iterations we'll optimize voting weights for
	'meta_learner_its':100,

	# what metric will we be using to evaluate?
	'metric':'f1',

	# base TPOT parameters
	'base_num_gens':2,
	'base_pop_size':2,
	'base_cv':2,

	# hidden TPOT parameters
	'hidden_num_gens':2,
	'hidden_pop_size':2,
	'hidden_cv':2,

	# voting TPOT parameters
	'voting_num_gens':5,
	'voting_pop_size':5,
	'voting_cv':5,

	# number of cv folds we use while training the whole ensemble
	'num_folds':2


}

base_models = {

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
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': range(2, 25),
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.2, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': range(2, 25),
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.2, 1.01, 0.05),
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
        'max_features': np.arange(0.2, 1.01, 0.05)
    },

	'sklearn.ensemble.AdaBoostClassifier': {
        'n_estimators': range(2,10),
        'learning_rate': np.arange(1e-3, 1.001, 1e-3),
    },    

    # 'sklearn.neighbors.KNeighborsClassifier': {
    #     'n_neighbors': range(1, 5),
    #     'weights': ["uniform", "distance"],
    #     'p': np.arange(2, 10, .1)
    # },

    # 'sklearn.svm.LinearSVC': {
    #     'penalty': ["l1", "l2"],
    #     'loss': ["hinge", "squared_hinge"],
    #     'dual': [True, False],
    #     'tol': np.arange(1e-5, 1e-1, 1e-4),
    #     'C': np.arange(1e-3, 1.001, 1e-3),
    #     'max_iter': [100]
    # },

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

    'xgboost.XGBClassifier': {
        'n_estimators': range(2,10),
        'max_depth': range(1, 11),
        'learning_rate': np.arange(1e-3, 1.001, 1e-3),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'nthread': [1]
    },

    # 'sklearn.linear_model.SGDClassifier': {
    #     'loss': ['log', 'hinge', 'modified_huber', 'squared_hinge', 'perceptron'],
    #     'penalty': ['elasticnet'],
    #     'alpha': [0.0, 0.001, 0.01, 0.1],
    #     'learning_rate': ['invscaling', 'constant'],
    #     'fit_intercept': [True, False],
    #     'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
    #     'eta0': [0.1, 1.0, 0.01],
    #     'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    # },

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
        'threshold': np.arange(0.0, 1.01, 0.05)
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

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
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

    'tpot.builtins.OneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False],
        'threshold': [10]
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
        'step': np.arange(0.05, .5, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': range(2,100),
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.2, 1.01, 0.05)
            }
        }
    },

    'sklearn.feature_selection.SelectFromModel': {
        #'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': range(2,100),
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.2, 1.01, 0.05)
            }
        }
    }
}

hidden_models = {
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
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': range(10, 50),
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.2, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': range(10, 50),
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.2, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': range(10, 50),
        'learning_rate': np.arange(1e-3, 1.001, 1e-3),
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.2, 1.01, 0.05)
    },

	'sklearn.ensemble.AdaBoostClassifier': {
        'n_estimators': range(10, 50),
        'learning_rate': np.arange(1e-3, 1.001, 1e-3),
    },    

    # 'sklearn.neighbors.KNeighborsClassifier': {
    #     'n_neighbors': range(1, 10),
    #     'weights': ["uniform", "distance"],
    #     'p': np.arange(2, 10, .1)
    # },

    # 'sklearn.svm.LinearSVC': {
    #     'penalty': ["l1", "l2"],
    #     'loss': ["hinge", "squared_hinge"],
    #     'dual': [True, False],
    #     'tol': np.arange(1e-5, 1e-1, 1e-4),
    #     'C': np.arange(1e-3, 1.001, 1e-3),
    #     'max_iter': [500]
    # },

    'sklearn.svm.SVC': {
        'tol': np.arange(1e-5, 1e-1, 1e-4),
        'C': np.arange(1e-3, 1.001, 1e-3),
        'max_iter': range(10, 5000),
        'probability': [True]
    },

    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': np.arange(1e-5, 1, 1e-4),
        'dual': [True, False]
    },

    'xgboost.XGBClassifier': {
        'n_estimators': range(10, 50),
        'max_depth': range(1, 11),
        'learning_rate': np.arange(1e-3, 1.001, 1e-3),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'nthread': [1]
    },

    # 'sklearn.linear_model.SGDClassifier': {
    #     'loss': ['log', 'hinge', 'modified_huber', 'squared_hinge', 'perceptron'],
    #     'penalty': ['elasticnet'],
    #     'alpha': [0.0, 0.001, 0.01, 0.1],
    #     'learning_rate': ['invscaling', 'constant'],
    #     'fit_intercept': [True, False],
    #     'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
    #     'eta0': [0.1, 1.0, 0.01],
    #     'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    # },

    'sklearn.neural_network.MLPClassifier': {
        'hidden_layer_sizes':[(10,), (20,), (30,), (40,), (50,), (60,), (70,), (80,), (90,), (100,), 
        					(10,5), (20,10), (30,15), (40,20), (50,25), (60,30), (70,35), (80,40), (90,45), (100,50)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.0, 0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': range(100, 500)
    },

    # Preprocesssors
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
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

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
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

    'tpot.builtins.OneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False],
        'threshold': [10]
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

    # 'sklearn.feature_selection.VarianceThreshold': {
    #     'threshold': np.arange(1e-4, .05, 1e-4)
    # },

    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': range(2,100),
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.2, 1.01, 0.05)
            }
        }
    },

    'sklearn.feature_selection.SelectFromModel': {
        #'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': range(2,100),
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.2, 1.01, 0.05)
            }
        }
    }
}

voting_models = {
	
	# Classifiers

    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': range(100, 500),
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.2, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': range(100, 500),
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.2, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': range(100, 500),
        'learning_rate': np.arange(1e-3, 1.001, 1e-3),
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.2, 1.01, 0.05)
    },

	'sklearn.ensemble.AdaBoostClassifier': {
        'n_estimators': range(100, 500),
        'learning_rate': np.arange(1e-3, 1.001, 1e-3),
    },    

    'xgboost.XGBClassifier': {
        'n_estimators': range(100, 500),
        'max_depth': range(1, 11),
        'learning_rate': np.arange(1e-3, 1.001, 1e-3),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'nthread': [1]
    },


    # 'sklearn.neural_network.MLPClassifier': {
    #     'hidden_layer_sizes':[(10,), (20,), (30,), (40,), (50,), (60,), (70,), (80,), (90,), (100,), 
    #     					(10,5), (20,10), (30,15), (40,20), (50,25), (60,30), (70,35), (80,40), (90,45), (100,50)],
    #     'activation': ['identity', 'logistic', 'tanh', 'relu'],
    #     'solver': ['lbfgs', 'sgd', 'adam'],
    #     'alpha': [0.0, 0.0001, 0.001, 0.01, 0.1],
    #     'learning_rate': ['constant', 'invscaling', 'adaptive'],
    #     'max_iter': range(1, 10)
    # },

    # Preprocesssors
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
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

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
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

    'tpot.builtins.OneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False],
        'threshold': [10]
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

    # 'sklearn.feature_selection.VarianceThreshold': {
    #     'threshold': np.arange(1e-4, .05, 1e-4)
    # },

    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': range(2,100),
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.2, 1.01, 0.05)
            }
        }
    },

    'sklearn.feature_selection.SelectFromModel': {
        #'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': range(2,100),
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.2, 1.01, 0.05)
            }
        }
    }
}

