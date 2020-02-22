# this is where you're going to put all of the config everythings!


# a config for the run itself
config = {
	
	# number of 'hidden' layers. Probably not the right term
	# but these are the layers that are in between the base and the meta-learner
	'num_hidden_layers':(1, 1),

	# number of models in the base. This is a range
	'num_base':(5, 10),

	# number of models in hidden layers. This is a range
	'num_hidden':(5, 10),

	# number of voters. For now we'll have this as just an int
	'num_voters':5,

	# are we going to do feature elimination based on correlation?
	'correlation_feature_elimination':True,

	# how many iterations for Boruta to run
	'max_iter_boruta':50,

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
	'voting_num_gens':3,
	'voting_pop_size':5,
	'voting_cv':5,

	# number of cv folds we use while training the whole ensemble
	'num_folds':5


}

base_models = {

}

hidden_models = {

}

voting_models = {

}

