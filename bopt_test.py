import numpy as np 
import pandas as pd 
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np 
import tensorflow as tf
#from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from mlens.ensemble import SuperLearner
import featuretools as ft 
import featuretools.variable_types as vtypes 
from featuretools.primitives import make_agg_primitive, make_trans_primitive
from featuretools.variable_types import Numeric

'''
df = pd.read_csv("creditcard.csv")
df = df.drop(['Time','V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
df = df.dropna()
train_df = df.sample(frac=0.8, random_state=0)
test_df =  df.drop(train_df.index)
train_df = shuffle(train_df)
test_df = shuffle(test_df)
y_train = train_df.pop('Class').as_matrix()
y_test = test_df.pop('Class').as_matrix()
X_train = train_df.as_matrix()
X_test = test_df.as_matrix()
'''
# finally let's import the data
df = pd.read_csv("creditcard.csv")
df = df.drop(['Time'], axis=1)
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

# alright here is where we're going to want to cut down all the variables
feature_list = fetch_feature_list()

# now filter out only the feature list
X = feature_matrix[feature_list]
del feature_matrix
X = X.fillna(X.mean())
X = X*1.0 # convert all to float hopefully
y = df.pop('Class')
del df

from cfig_older import *
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# a function to create the pbounds dictionary for each model
def make_pbounds():
    
    # initialize an empty dictionary for all the pbounds of the models to go into
    pbounds_dict = {}
    
    # we should already have global variables for the config things
    # run through the models in model_config cause those are the ones we're actually using
    for model_name in model_config.keys():
        
        # initialize the model's pbounds dict
        model_pbounds = {}
        
        # check if the hparam's dictionary is empty. If so, we need to keep the pbounds dict empty
        ##### IF THE PBOUNDS DICT IS EMPTY, IT MEANS THE MODEL IS NOT TO BE OPTIMIZED, JUST TRAINED
        if model_config[model_name]['hyperparameters'] == {}:
            pbounds_dict[model_name] = model_pbounds
            continue
        
        # and go through all the hyperparameters in here
        for hparam_name in model_config[model_name]['hyperparameters'].keys():
            
            ##### WE WILL NEED SUPPORT TO ADD THE ALIAS HERE AND REMOVE IT AT THE COMPLETION OF OPTIMIZATION
            ##### WHEN WE ALLOW MULTIPLE MODELS FROM EACH TYPE
            
            if type(model_config[model_name]['hyperparameters'][hparam_name]) is not dict:
                continue
            
            # if 'range' is 'base', dig into the classification_models dict
            if model_config[model_name]['hyperparameters'][hparam_name]['range'] == 'base':
                
                model_pbounds[hparam_name] = classification_models[model_name]['hyperparameters'][hparam_name]['default_range']
            
            # else we're good to use this one
            else:
                model_pbounds[hparam_name] = model_config[model_name]['hyperparameters'][hparam_name]['range']
        
        # now we should have a completed model_pbounds, add it to pbounds_dict
        pbounds_dict[model_name] = model_pbounds
    
    # then return the pbounds_dict!
    return pbounds_dict

# function to import all the models
##### I have nooo idea if this will do it globally or if we'll need some help
def import_models():
    
    # loop through all the models in model_config and call their import statements written in classification_models
    for model_name in model_config.keys():
        import_statement = classification_models[model_name]['import_statement']
        
        exec(import_statement, globals())

# a function to determine which kind of error we'll be optimizing for
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

# we need a function that's going to create our whole fleet of base models
# we may need to change the config so that we have a reduced set of hyperparameters
# that we randomly create the models from
# ideally, we want these to be weak learners. Shallow models randomly set up
# we'll also be using the stacking classifier so we just need to put these into a list all good!
def create_base_model_set():
    
    from random import random
    import_models()
    
    # we need to make a list to store all the models!
    base_model_list = []
    
    
    for model_name in model_config.keys():
        
        # pull out the low and high number of models to create
        low_model_num = model_config[model_name]['num_in_base'][0]
        high_model_num = model_config[model_name]['num_in_base'][1]
        
        # loop through the (random) number of these random models we want to create
        for i in range(int((high_model_num - low_model_num)*random() + low_model_num)):
        
            # check if the hparam's dictionary is empty. If it is empty, we don't do any randomization!
            # Just go ahead and add the model to the list
            if model_config[model_name]['hyperparameters_for_base'] == {}:
                temp_str = 'base_model_list.append({}())'.format(model_name)
                exec(temp_str)
                continue

            # let's make the dictionary for this model's hparams
            hparam_dict = {}

            # and go through all the hyperparameters in here
            for hparam_name in model_config[model_name]['hyperparameters_for_base'].keys():
                
                # plainly set hparam that we're not changing
                if type(model_config[model_name]['hyperparameters_for_base'][hparam_name]) is not dict:
                    hparam_dict[hparam_name] = model_config[model_name]['hyperparameters_for_base'][hparam_name]
                    continue

                # if 'range' is 'base', dig into the classification_models dict
                if model_config[model_name]['hyperparameters_for_base'][hparam_name]['range'] == 'base':

                    hparam_range = classification_models[model_name]['hyperparameters'][hparam_name]['default_range']

                # else we're good to use this one
                else:
                    hparam_range = model_config[model_name]['hyperparameters_for_base'][hparam_name]['range']

                # next we need to select which values of these we're going to be using, randomly
                lower = hparam_range[0]
                upper = hparam_range[1]

                hparam_value = (upper - lower)*random() + lower

                # and now we need to check if it needs to be an int
                if classification_models[model_name]['hyperparameters'][hparam_name]['type'] == 'int':
                    hparam_value = int(hparam_value)

                hparam_dict[hparam_name] = hparam_value

            # then add the model with this hparam dict to the model_list
            temp_str = 'base_model_list.append({}(**hparam_dict))'.format(model_name)
            exec(temp_str)
    
    print('Base models:')
    print(base_model_list)
    print()
    
    return base_model_list

# now we have a list created of our base models!
# let's do basically the same process, but for the 'hidden' layers
# we'll just loop through the number of layers we need to create and BAM should have something
def create_hidden_model_layers():
    
    from random import random
    
    # this will be a list of (lists containing the models for one layer)
    hidden_list = []
    
    low_layer_num = config['num_hidden_layers'][0]
    high_layer_num = config['num_hidden_layers'][1]
    
    for layer_num in range(int((high_layer_num - low_layer_num)*random() + low_layer_num)):
        
        # we need to make a list to store all the models!
        layer_model_list = []


        for model_name in model_config.keys():

            # pull out the low and high number of models to create
            low_model_num = model_config[model_name]['num_in_mid'][0]
            high_model_num = model_config[model_name]['num_in_mid'][1]

            # loop through the (random) number of these random models we want to create
            for i in range(int((high_model_num - low_model_num)*random() + low_model_num)):

                # check if the hparam's dictionary is empty. If it is empty, we don't do any randomization!
                # Just go ahead and add the model to the list
                if model_config[model_name]['hyperparameters'] == {}:
                    temp_str = 'layer_model_list.append({}())'.format(model_name)
                    exec(temp_str)
                    continue

                # let's make the dictionary for this model's hparams
                hparam_dict = {}

                # and go through all the hyperparameters in here
                for hparam_name in model_config[model_name]['hyperparameters'].keys():
                    
                        
                    if type(model_config[model_name]['hyperparameters'][hparam_name]) is not dict:
                        hparam_dict[hparam_name] = model_config[model_name]['hyperparameters'][hparam_name]
                        continue
                    
                    
                    
                    # if 'range' is 'base', dig into the classification_models dict
                    if model_config[model_name]['hyperparameters'][hparam_name]['range'] == 'base':

                        hparam_range = classification_models[model_name]['hyperparameters'][hparam_name]['default_range']

                    # else we're good to use this one
                    else:
                        hparam_range = model_config[model_name]['hyperparameters'][hparam_name]['range']

                    # next we need to select which values of these we're going to be using, randomly
                    lower = hparam_range[0]
                    upper = hparam_range[1]

                    hparam_value = (upper - lower)*random() + lower

                    # and now we need to check if it needs to be an int
                    if classification_models[model_name]['hyperparameters'][hparam_name]['type'] == 'int':
                        hparam_value = int(hparam_value)

                    hparam_dict[hparam_name] = hparam_value
                

                # then add the model with this hparam dict to the model_list
                temp_str = 'layer_model_list.append({}(**hparam_dict))'.format(model_name)
                exec(temp_str)
        
        # then append this layer model list to the overall hidden list
        hidden_list.append(layer_model_list)
        
    print('Hidden models:')
    print(hidden_list)
    print()
    
    return hidden_list


# build the ensemble!
def build_ensemble(base_model_list, hidden_list):
    
    ens = SuperLearner(verbose=2) # lower this verbosity later if you don't want it
    ens.add(base_model_list)#, proba=True) # making probability on all these. We can talk about propagate_features later
    
    # then loop through all the other layers
    for layer in hidden_list:
        
        ens.add(layer, proba=True)
    
    return ens
    
    
# once we have all of these, we need to train everything except the meta-learner on this data.
# I hope that's possible to do that and just return the probabilities!
def train_layers(ens, X_train, y_train, X_test):
    
    print('Training base models')
    
    ### WE MAY NEED TO DO THIS ACROSS SEVERAL FOLDS!
    ens.fit(X_train, y_train)
    
    # we just want to return the prediction. This is hopefully one column for each model.
    # we want to train the metalearner on this set, with y_test, and have these train-test splitted.
    return ens.predict(X_test)


def error_function_meta(**hparams):
    
    
    model_name = model_name_temp_list[0] # I have no idea what I'm doing here
    
    # we need to go through each hparam in hparams and find out which ones need to be made into an int!
    for hparam_name in hparams.keys():

        if classification_models[model_name]['hyperparameters'][hparam_name]['type'] == 'int':
            hparams[hparam_name] = int(hparams[hparam_name])

    
    # then create the model
    # I surely hope we can do this in string format. May need to play around with global-ness
    # may also need some asterisks
    temp_list = []
    model_str = 'temp_list.append({}(**hparams))'.format(model_name)
    exec(model_str)
    model = temp_list[0]

    # fit the model
    model.fit(meta_x_train, meta_y_train)
    
    preds = model.predict(meta_x_test)
    
    
    try:
        temp_error = error(preds, meta_y_test)
    except:
        temp_error = 0

    return temp_error

# what the heck am I doing
model_name_temp_list = []

# then the meta learner
def run_meta_optimization(ens):
    
    # then make the pbounds dictionary
    pbounds_dict = make_pbounds()
    
    # now we need to get a list of models that we actually want to optimize
    meta_model_name_list = []
    for meta_model_name in model_config.keys():
        
        if model_config[meta_model_name]['metalearner']:
            
            meta_model_name_list.append(meta_model_name)
    
    # and then loop through each one of these models we'll be trying!
    max_target = 0
    max_meta_dict = {}
    for meta_model_name in meta_model_name_list:
        
        print('Optimizing meta learner {}'.format(meta_model_name))
        
        
        ##### we should add some time element in here too, that would be cool.

        # get the pbounds
        model_pbounds = pbounds_dict[meta_model_name]
        
        # ugh we need to set the name and everything
        #exec('model_name = {}'.format(meta_model_name), globals())
        if(len(model_name_temp_list) == 0):
            model_name_temp_list.append(meta_model_name)
        else:
            model_name_temp_list[0] = meta_model_name

        optimizer = BayesianOptimization(
            f=error_function_meta,
            pbounds=model_pbounds)
        optimizer.maximize(init_points=2, n_iter=config['meta_learner_its'], xi=0.5)
        max_hparams = optimizer.max
        
        target = max_hparams['target']
        
        print()
        print(' Finished {} optimization. Target: {}'.format(meta_model_name, target))
        
        if target > max_target:
            
            print('  New max target found!')
            
            # reassign the target
            max_target = target
            
            # then update the max_meta_dict
            max_meta_dict = {}
            max_meta_dict[meta_model_name] = max_hparams
    
    
    max_meta_dict['layers'] = ens
    
    print()
    print('Finished entire optimization')
    print('Max model found, with target {}:'.format(max_target))
    print(max_meta_dict)    
    
def run():
    
    run_meta_optimization(hparam_max_dict)

from config import *
from tpot import TPOTClassifier
import warnings
warnings.filterwarnings('ignore')

base_model_list = create_base_model_set()
hidden_list = create_hidden_model_layers()
ens = build_ensemble(base_model_list, hidden_list)
ens_preds = train_layers(ens, X_train, y_train, X_test)
meta_x_train, meta_x_test, meta_y_train, meta_y_test = train_test_split(ens_preds, y_test, train_size=0.75, test_size=0.25)

'''
tpot = TPOTClassifier(verbosity=2, 
                      scoring="f1",  
                      n_jobs=-1, 
                      generations=100, 
                      population_size=100)


tpot = tpot.fit(X_train, y_train)
s = tpot.score(X_test, y_test)

print('')
print('Validation score is: {}'.format())
'''
run_meta_optimization(ens)


