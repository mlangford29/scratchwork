
# this is a function to do a search of important features across a list of primitives

##### also idea!
#####  what if you had a whole list of the primitives you want to test
#####  go through each of the primitives one at a time
#####  boostaroota that
#####  see what the most important features are from that
#####  then generate a list of the most important features
#####   and primitives that had them
#####  save those features and their importances
#####  So you'll need to do the boostaroota step and then xgboost right after to grab the importances
#####  That will be in a dictionary format and you can add those pretty easily!
#####  from that dictionary you can save it!

import featuretools as ft 
import featuretools.variable_types as vtypes 
from featuretools.primitives import make_agg_primitive, make_trans_primitive
from featuretools.variable_types import Numeric
from boostaroota import BoostARoota
import pandas as pd
import xgboost as xgb
import numpy as np

# importance threshold to remove a feature. If it's below this then we won't use it
threshold = 5

# number of iterations to fit the model that actually calculates feature importance
num_its = 20

df = pd.read_csv("creditcard.csv")
df = df.drop(['Time'], axis=1)
df = df.dropna()


# ok and then we'll do all the featuretools things that need to happen
es = ft.EntitySet(id = 'card') # no clue what this means but whatever

# make an entity from the observations data
es = es.entity_from_dataframe(dataframe = df.drop('Class', axis=1),
								entity_id = 'obs',
								index = 'index')

y = df.pop('Class')

##### ARE WE ALSO GOING TO TEST SOME OF OUR OWN?
##### if we can get manhattan distance or something set up that would be cool
# these are the trans primitives we're going to test
trans_primitive_list = ['divide_by_feature', 'add_numeric', 'cum_mean', 'not_equal', 
	'cum_sum', 'equal', 'less_than_equal_to', 
	'multiply_boolean', 'greater_than_equal_to_scalar', 
	'multiply_numeric', 'diff',  
	'modulo_numeric_scalar', 'subtract_numeric_scalar', 
	'divide_numeric_scalar', 'add_numeric_scalar', 
	'subtract_numeric', 'cum_min', 'not_equal_scalar', 'cum_count', 
	'equal_scalar', 'divide_numeric',  
	'percentile', 'greater_than', 'less_than', 'multiply_numeric_scalar', 
	'greater_than_equal_to', 'modulo_by_feature', 'scalar_subtract_numeric_feature', 
	'isin', 'absolute', 'modulo_numeric']



# empty dictionary to store all the features and their importances
total_feat_imp = {}

# empty list to store the useful trans primitives
useful_prim_list = []

for t_prim in trans_primitive_list:

	print()
	print('Testing {}'.format(t_prim))

	df_, feature_names = ft.dfs(entityset=es, target_entity='obs',
										#agg_primitives = ['min', 'max', 'mean', 'count', 'sum', 'std', 'trend'],
										trans_primitives = [t_prim],
										max_depth=1,
										n_jobs=1,
										verbose=0)

	##### CHECK IF ANY OF THE COLUMNS HAVE INF IN THEM
	##### ADD TO A LIST
	##### DROP THAT SHIT
	inf_list = []
	for feature in list(df_.columns):

		if (np.inf in df_[feature].to_numpy()) or (np.NINF in df_[feature].to_numpy()):
			inf_list.append(feature)

	if len(inf_list) > 0:
		print(' Dropping {} features for containing inf'.format(inf_list))
		df_.drop(inf_list, axis=1)

	# would it go faster if we can drop all the original columns?
	df_ = df_.drop(['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
					'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
					'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'], axis=1)

	df_ = df_.dropna(how='any', axis=1)
	X = df_ # and another copy. Might not need this



	print()
	print('Starting Boruta')

	br = BoostARoota(metric='logloss', silent=True) #cutoff=.5, delta=0.7, silent=True)
	br.fit(X, y)
	chosen_features = br.keep_vars_

	if len(chosen_features) == 0:
		print(' No chosen features!')
		continue

	ind = range(len(chosen_features))

	# let's do multiple rounds of this fitting and average
	avg_dict = {}

	for i in range(num_its):

		xgb_for_feat_imp = xgb.train(dtrain = xgb.DMatrix(X[chosen_features], label=y), params={})
		ft_imp_dict = xgb_for_feat_imp.get_score(importance_type='gain')

		for key in ft_imp_dict.keys():

			if key not in avg_dict.keys():

				avg_dict[key] = ft_imp_dict[key]

			else:

				avg_dict[key] += ft_imp_dict[key]

	# delete feature importance dict, remake in a minute
	ft_imp_dict = {}

	# then average it out
	for key in avg_dict.keys():

		ft_imp_dict[key] = avg_dict[key]/num_its

	# we need to check if no chosen feature has importance above a threshold
	# so maybe we'll remove keys if their value is below the threshold
	# add the deleted keys to a list, remove later
	bad_key_list = []

	for key in ft_imp_dict.keys():

		if ft_imp_dict[key] < threshold:
			bad_key_list.append(key)

	# now we should have a list of the bad ones
	# go through them all and delete the bad ones
	for bad_key in bad_key_list:
		del ft_imp_dict[bad_key]

	# you should do another check to see if the number of features is equal to what it was originally
	# if it is, then that means no additional feature was important
	# if it's not (and there were additional important features)
	#  then you need to add this to a list of trans primitives that you should use!
	if len(list(ft_imp_dict.keys())) > 0:
		useful_prim_list.append(t_prim)

	# now you have a dictionary of the importances from that single trans primitive
	# we can go through all the keys in this and add them!
	for feat in ft_imp_dict.keys():

		if feat not in total_feat_imp.keys():
			total_feat_imp[feat] = ft_imp_dict[feat]


ft_imps = pd.DataFrame.from_dict(total_feat_imp, orient='index', columns=['importance']).sort_values('importance', ascending=False)
ft_imps.to_csv('feature_search.csv')

print()
print('Final list of primitives to use:')
print(useful_prim_list)

'''
['add_numeric', 'less_than_scalar', 'less_than_equal_to', 'greater_than_equal_to_scalar', 'multiply_numeric', 'greater_than_scalar', 'subtract_numeric_scalar', 'divide_numeric_scalar', 'add_numeric_scalar', 'divide_by_feature', 'subtract_numeric', 'divide_numeric', 'less_than_equal_to_scalar', 'percentile', 'greater_than', 'less_than', 'multiply_numeric_scalar', 'greater_than_equal_to', 'modulo_by_feature', 'scalar_subtract_numeric_feature', 'absolute', 'modulo_numeric']
'''

