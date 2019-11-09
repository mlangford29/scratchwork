import math
import numpy as np
import random
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

### user definitions
n_train_pts = 1000
n_test_pts = 200 # number of test points to gather from each test region

# a range of x values to use across all functions for training
x_min = -10
x_max = 10

dataset_list = ['linear', 'quadratic', 'abs', 'sinusoid', 'exponential']

### end user definitions

### dataset selection

# we'll create a different function for each one of these
# for each function, input an array of x values
# it'll return the y array for that dataset. They'll all have the same x array
# this will be put into a dictionary by a separate function
def linear(x):
	return x

def quadratic(x):
	return x**2

def abs(x):
	return np.absolute(x)

def sinusoid(x):
	return np.sin(x)

def exponential(x):
	return np.exp(x)

### end dataset selection

### model selection

# create a dictionary to store all the models in
model_dict = {}

# list of models, with parameters
model_dict['XGB'] = GradientBoostingRegressor()
model_dict['SupportVector'] = SVR()
model_dict['RandomForest'] = RandomForestRegressor()
model_dict['ExtraTrees'] = ExtraTreesRegressor()
model_dict['AdaBoost'] = AdaBoostRegressor()
model_dict['KNN'] = KNeighborsRegressor()
model_dict['RadiusNeighbors'] = RadiusNeighborsRegressor()
model_dict['NN'] = MLPRegressor()
### end model selection

# function to create the dataset dictionary
def trainset_creation():

	# create a numpy x array
	x = np.array()
	for _ in range(n_train_pts):
		x.append(random.random()*(x_max - x_min) + x_min)

	# now create the dictionary to store the datasets
	trainset_dict = {}

	for dataset_name in dataset_list:

		y = exec('{}(x)'.format(dataset_name))

		trainset_dict[dataset_name] = [x, y]

	return trainset_dict

def testset_creation():

	# need to create the x ranges for all the types we need
	# maybe we'll split the number of points between the positive and negative sides
	# get equal numbers on each side

	# first, let's create the in_bounds x set
	x_in_bounds = np.array()
	for _ in range(n_test_pts):
		x_in_bounds.append(random.random()*(x_max - x_min) + x_min)

	# now one for the 10% out of bounds
	# we'll need 2 sets of bounds. 
	# x_min_10 is 10% lower than x_min. x_min will be the upper bound of the lower range
	# x_max_10 is 10% higher than x_max. x_max will be the lower bound of the upper range
	x_min_10 = x_min - math.abs(x_min*0.1)
	x_max_10 = x_max + math.abs(x_max*0.1)

	# create the set for that
	x_10 = np.array()

	# need to split this process in two
	for _ in range(n_test_pts/2):
		x_10.append(random.random()*(x_min - x_min_10) + x_min_10)
	for _ in range(n_test_pts/2):
		x_10.append(random.random()*(x_max_10 - x_max) + x_max)

	# cool, now repeat the process for 50%
	x_min_50 = x_min - math.abs(x_min*0.5)
	x_max_50 = x_max + math.abs(x_max*0.5)
	x_50 = np.array()
	for _ in range(n_test_pts/2):
		x_50.append(random.random()*(x_min - x_min_50) + x_min_50)
	for _ in range(n_test_pts/2):
		x_50.append(random.random()*(x_max_50 - x_max) + x_max)

	# and 100
	x_min_100 = 2*x_min
	x_max_100 = 2*x_max 
	x_100 = np.array()
	for _ in range(n_test_pts/2):
		x_100.append(random.random()*(x_min - x_min_100) + x_min_100)
	for _ in range(n_test_pts/2):
		x_100.append(random.random()*(x_max_10 - x_max) + x_max)

	# cool, now that we have all the x sets for testing, we need to create the y sets and dictionize
	testset_dict = {}

	for dataset_name in dataset_list:

		# no easy loop for this, sorry
		#dict_in_bounds = {}
		dataset_temp_dict = {}
		y = exec('{}(x_in_bounds)'.format(dataset_name))
		dataset_temp_dict['in_bounds'] = [x_in_bounds, y]

		#dict_10 = {}
		y = exec('{}(x_10)'.format(dataset_name))
		dataset_temp_dict['10_percent_out'] = [x_10, y]

		#dict_50 = {}
		y = exec('{}(x_50)'.format(dataset_name))
		dataset_temp_dict['50_percent_out'] = [x_50, y]

		#dict_100 = {}
		y = exec('{}(x_100)'.format(dataset_name))
		dataset_temp_dict['100_percent_out'] = [x_100, y]

		testset_dict[dataset_name] = dataset_temp_dict

	return testset_dict




		




