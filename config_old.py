# this is where you're going to put all of the config everythings!


# a config for the run itself
# we'll assume the model_config is true! Thanks
config = {
	
	# number of bayesian opt iterations we'll train each model for
	'num_model_its':5,
	'meta_learner_its':50,
	'metric':'f1',

	# number of sub-iterations we're going to train the model for
	# for robustness of metric optimization
	'num_sub_its':3

}

# more thoughts
# what if we want to lock down a hyperparameter at a value
# we're not trying to optimize it, but we just don't want to use the default sklearn value?
# maybe we'll have a key in the hyperparameter of "value"
# then we'll check to see if value is present in the hparam's keys
# if it is, then we need to just use that value and we're not optimizing for it
# we'll just add support for that but not necessarily use it right now

# then a dictionary for the models we'll be using and the actual ranges we'll use
# and which hyperparameters we'll be tuning for each one of the models
model_config = {
	

#	'XGBoost':{
#
#		##### LATER ADD SUPPORT FOR NUMBER OF THESE MODELS TO USE
#
#		'hyperparameters':{
#			'n_estimators':{
#
#				# for these we'll just have 'range' or 'value'
#				# if the hparam isn't in here, its default sklearn value will be used instead
#				'range':(10, 500)
#			},
#			'eta':{
#				# we'll also say 'base' if we want to just use what's in "classification_models"
#				'range':'base'
#			},
#			'max_depth':{
#				'range':(1, 6)
#			},
#			'subsample':{
#				'range':'base'
#			},
#		}
#	},

	'LogisticRegression':{
		'hyperparameters':{
			'tol':{
				'range':'base'
			},
			'C':{
				'range':'base'
			}
		}
	},

#	'KNeighborsClassifier':{
#		'hyperparameters':{
#			'n_neighbors':{
#				'range':(2, 100)
#			},
#			'p':{
#				'range':(1, 5)
#			}
#		}
#	},

	'GaussianNB':{
		'hyperparameters':{}
	},

#	'SVC':{
#		'hyperparameters':{
#			'C':{
#				'range':'base'
#			},
#			'tol':{
#				'range':'base'
#			}
#		}
#	},

	'ExtraTreesClassifier':{
		'hyperparameters':{
			'n_estimators':{
				'range':(10, 500)
			},
			'max_depth':{
				'range':(2, 10)
			}
		}
	},

#	'RandomForestClassifier':{
#		'hyperparameters':{
#			'n_estimators':{
#				'range':(10, 500)
#			},
#			'max_depth':{
#				'range':(1, 10)
#			},
#		}
#	}
#
	'AdaBoostClassifier':{
		'hyperparameters':{
			'n_estimators':{
				'range':(10, 500)
			},
			'learning_rate':{
				'range':'base'
			}
		}
	}
}



# config to place all of the models we *can* use
# for now, just classification
classification_models = {
	
	'XGBoost':{
		'alias':'xgb',
		'import_statement':'import XGBoost',
		'hyperparameters':{

			# in here we should include what the default or recommended range is for the optimization
			'n_estimators':{
				'type':'int',
				'default_range':(5, 500)
			},
			'eta':{
				'type':'float',
				'default_range':(0.0001, 0.5)
			},
			'max_depth':{
				'type':'int',
				'default_range':(1, 12)
			},
			'subsample':{
				'type':'float',
				'default_range':(0.01, 0.99)
			},
			'scale_pos_weight':{
				'type':'float',
				'default_range':(1, 10000)
			}
		}
	},

	'LogisticRegression':{
		'alias':'lr',
		'import_statement':'from sklearn.linear_model import LogisticRegression',
		'hyperparameters':{

			'tol':{
				'type':'float',
				'default_range':(0.000001, 0.01)
			},
			'C':{
				'type':'float',
				'default_range':(0.00001, 10)
			}
		}
	},

	'KNeighborsClassifier':{
		'alias':'knn',
		'import_statement':'from sklearn.neighbors import KNeighborsClassifier',
		'hyperparameters':{

			'n_neighbors':{
				'type':'int',
				'default_range':(2, 100)
			},
			'p':{
				'type':'int',
				'default_range':(1, 5)
			}
		}
	},

	'GaussianNB':{
		'alias':'nb',
		'import_statement':'from sklearn.naive_bayes import GaussianNB',
		##### there are no hyperparameters for NB!
		# we can probably hard code this one doesn't need to be optimized, just used
		'hyperparameters':{}
	},

	'SVC':{
		'alias':'svc',
		'import_statement':'from sklearn.svm import SVC',
		'hyperparameters':{

			'C':{
				'type':'float',
				'default_range':(0.1, 20)
			},
			'tol':{
				'type':'float',
				'default_range':(0.00001, 0.001)
			},
			'max_iter':100000
		}
	},

	'ExtraTreesClassifier':{

		'alias':'etc',
		'import_statement':'from sklearn.ensemble import ExtraTreesClassifier',
		'hyperparameters':{
			'n_estimators':{
				'type':'int',
				'default_range':(5, 500)
			},
			'max_depth':{
				'type':'int',
				'default_range':(2, 12)
			},
			'scale_pos_weight':{
				'type':'float',
				'default_range':(1, 10000)
			}
		}
	},

	'RandomForestClassifier':{

		'alias':'rfc',
		'import_statement':'from sklearn.ensemble import RandomForestClassifier',
		'hyperparameters':{

			# in here we should include what the default or recommended range is for the optimization
			'n_estimators':{
				'type':'int',
				'default_range':(1,500)
			},
			'max_depth':{
				'type':'int',
				'default_range':(1, 12)
			}
		}
	},

	'AdaBoostClassifier':{

		'alias':'ada',
		'import_statement':'from sklearn.ensemble import AdaBoostClassifier',
		'hyperparameters':{

			# in here we should include what the default or recommended range is for the optimization
			'learning_rate':{
				'type':'float',
				# changing this one because there's a trade-off between learning rate and n_est
				'default_range':(0.01, 10)
			},
			'n_estimators':{
				'type':'int',
				'default_range':(1,500)
			}
		}
	}
}



