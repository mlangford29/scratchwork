# this is where you're going to put all of the config everythings!


# a config for the run itself
# we'll assume the model_config is true! Thanks
config = {
	
	# number of 'hidden' layers. Probably not the right term
	# but these are the layers that are in between the base and the meta-learner
	'num_hidden_layers':(1, 1),

	# number of bayesian opt iterations we'll train each model for
	'meta_learner_its':100,
	'metric':'f1',

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
##### MAYBE HAVE SUPPORT FOR 'LEARNER' BOOL SO THAT YOU CAN EXCLUDE SOME FROM BEING BASE LEARNERS
model_config = {

	'LogisticRegression':{
		'metalearner':True,

		# RANGE DOES NOT INCLUDE UPPER BOUND
		'num_in_base':(0, 10),
		'num_in_mid':(0, 5),
		'hyperparameters_for_base':{
			'tol':{
				'range':'base'
			},
			'C':{
				'range':'base'
			}
		},

		# these are the hyperparameter ranges for when they're not a base model
		'hyperparameters':{
			'tol':{
				'range':'base'
			},
			'C':{
				'range':'base'
			}
		}
	},

	'KNeighborsClassifier':{
		'metalearner':False,
		'num_in_base':(0, 10),
		'num_in_mid':(0, 5),
		'hyperparameters_for_base':{
			'n_neighbors':{
				'range':(1, 20)
			},
			'p':{
				'range':(1, 5)
			}
		},
		'hyperparameters':{
			'n_neighbors':{
				'range':(1, 100)
			},
			'p':{
				'range':(1, 5)
			}
		}
	},

	'GaussianNB':{
		'metalearner':False,
		'num_in_base':(0, 2),
		'num_in_mid':(0, 2),
		'hyperparameters_for_base':{},
		'hyperparameters':{}
	},

	'SVC':{
		'metalearner':False,
		'num_in_base':(0, 10),
		'num_in_mid':(0, 5),
		'hyperparameters_for_base':{
			'C':{
				'range':'base'
			},
			'tol':{
				'range':'base'
			},
			'probability':True,
			'max_iter':1000
		},
		'hyperparameters':{
			'C':{
				'range':'base'
			},
			'tol':{
				'range':'base'
			},
			'probability':True
		}
	},

	'ExtraTreesClassifier':{
		'metalearner':True,
		'num_in_base':(0, 10),
		'num_in_mid':(0, 5),
		'hyperparameters_for_base':{
			'n_estimators':{
				'range':(2, 20)
			},
			'max_depth':{
				'range':(1, 5)
			}
		},
		'hyperparameters':{
			'n_estimators':{
				'range':(10, 500)
			},
			'max_depth':{
				'range':(2, 10)
			}
		}
	},

	'RandomForestClassifier':{
		'metalearner':True,
		'num_in_base':(0, 10),
		'num_in_mid':(0, 5),
		'hyperparameters_for_base':{
			'n_estimators':{
				'range':(2, 20)
			},
			'max_depth':{
				'range':(1, 5)
			},
		},
		'hyperparameters':{
			'n_estimators':{
				'range':(10, 500)
			},
			'max_depth':{
				'range':(1, 10)
			},
		}
	},

	'AdaBoostClassifier':{
		'metalearner':True,
		'num_in_base':(0, 10),
		'num_in_mid':(0, 5),
		'hyperparameters_for_base':{
			'n_estimators':{
				'range':(2, 20)
			},
			'learning_rate':{
				'range':'base'
			}
		},
		'hyperparameters':{
			'n_estimators':{
				'range':(10, 500)
			},
			'learning_rate':{
				'range':'base'
			}
		}
	},

	# 'Lasso':{
	# 	'metalearner':False,
	# 	'num_in_base':(0, 10),
	# 	'num_in_mid':(0, 5),
	# 	'hyperparameters_for_base':{
	# 		'alpha':{
	# 			'range':'base'
	# 		},
	# 		'tol':{
	# 			'range':'base'
	# 		}
	# 	},
	# 	'hyperparameters':{
	# 		'alpha':{
	# 			'range':'base'
	# 		},
	# 		'tol':{
	# 			'range':'base'
	# 		}
	# 	}
	# },

	# 'Ridge':{
	# 	'metalearner':False,
	# 	'num_in_base':(0, 10),
	# 	'num_in_mid':(0, 5),
	# 	'hyperparameters_for_base':{
	# 		'alpha':{
	# 			'range':'base'
	# 		},
	# 		'tol':{
	# 			'range':'base'
	# 		}
	# 	},
	# 	'hyperparameters':{
	# 		'alpha':{
	# 			'range':'base'
	# 		},
	# 		'tol':{
	# 			'range':'base'
	# 		}
	# 	}
	# },

	# 'ElasticNet':{
	# 	'metalearner':False,
	# 	'num_in_base':(0, 10),
	# 	'num_in_mid':(0, 5),
	# 	'hyperparameters_for_base':{
	# 		'alpha':{
	# 			'range':'base'
	# 		},
	# 		'l1_ratio':{
	# 			'range':'base'
	# 		}
	# 	},
	# 	'hyperparameters':{
	# 		'alpha':{
	# 			'range':'base'
	# 		},
	# 		'l1_ratio':{
	# 			'range':'base'
	# 		}
	# 	}
	# },

#	'GaussianProcessClassifier':{
#		'metalearner':True,
#		'num_in_base':(0, 5),
#		'num_in_mid':(0, 3),
#		'hyperparameters_for_base':{
#			'max_iter_predict':{
#				'type':'int',
#				'range':'base'
#			},
#			'n_restarts_optimizer':{
#				'type':'int',
#				'range':'base'
#			}
#		},
#		'hyperparameters':{
#
#			'max_iter_predict':{
#				'type':'int',
#				'range':'base'
#			},
#			'n_restarts_optimizer':{
#				'type':'int',
#				'range':'base'
#			}
#		}
#	},

	'DecisionTreeClassifier':{
		'metalearner':False,
		'num_in_base':(0, 50),
		'num_in_mid':(0, 5),
		'hyperparameters_for_base':{
			'max_depth':{
				'type':'int',
				'range':'base'
			}
		},
		'hyperparameters':{

			'max_depth':{
				'type':'int',
				'range':'base'
			}

			# maybe have a categorical variable for criterion
		}
	}

#	'MLPClassifier':{
#		'metalearner':True,
#		'num_in_base':(0, 5),
#		'num_in_mid':(0, 3),
#		'hyperparameters_for_base':{
#			'n_estimators':{
#				'range':(2, 20)
#			},
#			'learning_rate':{
#				'range':'base'
#			}
#		},
#		'hyperparameters':{
#
#			# not going to use ACTUAL hyperparameters here
#			# this will require some redoing in the actual script to make it work
#
#			'num_layers':{
#				'type':'int',
#				'default_range':(1, 10)
#			},
#
#			'hidden_layer_size':{
#				'type':'int',
#				'default_range':(2, 100)
#			},
#
#			'max_iter':{
#				'type':'int',
#				'default_range':(1, 10)
#			}
#		}
#	}
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
			'probability':True,
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

			'learning_rate':{
				'type':'float',
				# changing this one because there's a trade-off between learning rate and n_est
				'default_range':(0.001, 2)
			},
			'n_estimators':{
				'type':'int',
				'default_range':(1,500)
			}
		}
	},

	'Lasso':{

		'alias':'lasso',
		'import_statement':'from sklearn.linear_model import Lasso',
		'hyperparameters':{

			'alpha':{
				'type':'float',
				'default_range':(0.01, 2)
			},
			'tol':{
				'type':'float',
				'default_range':(.00001,.01)
			}
		}
	},

	'Ridge':{

		'alias':'ridge',
		'import_statement':'from sklearn.linear_model import Ridge',
		'hyperparameters':{

			'alpha':{
				'type':'float',
				'default_range':(0.01,2)
			},
			'tol':{
				'type':'float',
				'default_range':(.00001,.01)
			}
		}
	},

	'ElasticNet':{

		'alias':'enet',
		'import_statement':'from sklearn.linear_model import ElasticNet',
		'hyperparameters':{

			'alpha':{
				'type':'float',
				'default_range':(0.01,2)
			},
			'l1_ratio':{
				'type':'float',
				'default_range':(.1,.9)
			}
		}
	},

	'GaussianProcessClassifier':{

		'alias':'gpc',
		'import_statement':'from sklearn.gaussian_process import GaussianProcessClassifier',
		'hyperparameters':{

			'max_iter_predict':{
				'type':'int',
				'default_range':(10,1000)
			},
			'n_restarts_optimizer':{
				'type':'int',
				'default_range':(0,5)
			},
		}
	},

	'DecisionTreeClassifier':{

		'alias':'dtc',
		'import_statement':'from sklearn.tree import DecisionTreeClassifier',
		'hyperparameters':{

			'max_depth':{
				'type':'int',
				'default_range':(1, 10)
			}

			# maybe have a categorical variable for criterion
		}
	}

#	'MLPClassifier':{
#
#		'alias':'mlp',
#		'import_statement':'from sklearn.neural_network import MLPClassifier',
#		'hyperparameters':{
#
#			# not going to use ACTUAL hyperparameters here
#			# this will require some redoing in the actual script to make it work
#
#			'num_layers':{
#				'type':'int',
#				'default_range':(1, 10)
#			},
#
#			'hidden_layer_size':{
#				'type':'int',
#				'default_range':(2, 100)
#			},
#
#			'max_iter':{
#				'type':'int',
#				'default_range':(1, 10)
#			}
#		}
#	}

}



