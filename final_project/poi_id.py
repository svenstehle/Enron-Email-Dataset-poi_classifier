#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.stats import pearsonr
import time
#from sklearn_pandas import DataFrameMapper
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, StratifiedShuffleSplit, cross_val_score
#from sklearn.decomposition import PCA, NMF 
    # Non-Negative Matrix Factorization (NMF)
    # Find two non-negative matrices (W, H) whose product approximates the non- negative matrix X. 
    # This factorization can be used for example for dimensionality reduction, source separation or topic extraction.
from sklearn import svm
from sklearn import linear_model, neighbors, ensemble
from sklearn.linear_model import LogisticRegression
import xgboost

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn import model_selection, feature_selection, metrics
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, precision_recall_curve, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import f_classif, SelectKBest # feature extraction with Statistical Selection

from xgboost.sklearn import XGBClassifier
import matplotlib.pylab as pyl
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 8
plt.style.use("ggplot")
sns.set_style("white")
from sklearn.naive_bayes import GaussianNB

import sys
import pickle
#%%
#sys.path.append("D:/Laptop Transfer/Udemy/Udacity intro to Machine Learning/tools/")
#sys.path.append("D:/Laptop Transfer/Udemy/Udacity intro to Machine Learning/final_project/")

sys.path.append("E:/datasets/Enron Data/tools/")
sys.path.append("E:/datasets/Enron Data/final_project/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
#Identify Fraud from Enron Email
# ML can help us identify patterns that we would otherwise not find easily or at all with just human intuition
# We can therefore identify pois that haven't been indicted or made public otherwise

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features

# we will use all the features and iteratively select those that give us the most informative value, e.g. higher 
# classification scores (precision and recall, since this is an imbalanced classification task)


### Load the dictionary containing the dataset
#with open("D:/Laptop Transfer/Udemy/Udacity intro to Machine Learning/final_project/final_project_dataset.pkl", "rb") as data_file:
with open("E:/datasets/Enron Data/final_project/final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


### Store to my_dataset for easy export below.
my_dataset = data_dict
#%%
pd.set_option("display.max_columns",22)
pd.set_option("display.max_rows",15)
#%%
df = pd.DataFrame(my_dataset).T
df.sample(10)
# first we have to clean the data, get rid of "objects" in every column, since we imported everything as string data
# columns with numbers: bonus, deferral_payments, deferred_income, director_fees, exercised_stock_options, expenses, from_messages,
# from_poi_to_this_person, from_this_person_to_poi, loan_advances, long_term_incentive, other, restricted_stock, restricted_stock_deferred,
# salary, shared_receipt_with_poi, to_messages, total_payments, total_stock_value
# columns with string data: email_address
# target (bool, convert to 0/1): poi
# ok so the most elegant way to go about this that I can come up with right now is:
# - transform all the columns except email_address and poi with one function, write a function for poi and email_address separately
#%%
target = "poi"

def datacleaning(data, target, skip = "email_address"):
    predictors = data.columns
    for col in predictors:
        if col == target: # transform true and false into binary
            df[col] = df[col].map(lambda x: 1 if x else 0).astype(np.int)
        elif col == skip: # transform nan into np.nan and everything else into strings
            df[col] = df[col].map(lambda x: np.nan if x == 'NaN' else x)
        else: # transform nan into np.nan and everything else into floats
            df[col] = df[col].map(lambda x: np.nan if x == 'NaN' else x).astype(np.float)
    return df
#%%
df = datacleaning(df, target)    
#%%
df.sample(10)
#%%
#are there features with many missing values? --> yes, a lot
for col in df.columns:
    if np.any(pd.isnull(df[col])): # check every column for nan and then check how many nan we have in that column
        num_nan_values = pd.isnull(df[col]).value_counts()[1] # True is the second entry in value_counts()
        print("{} nan\tvalues in col {}!".format(num_nan_values,col)) 

#create a train set to play around with
train = df.copy(deep = True)
#%%
#impute nan with -1
train["email_address"].fillna(value = -1, inplace = True)
#%%
train["email_address"].value_counts()
#%%
# label encode the email_address:
label = LabelEncoder()
train["email_address_code"] = label.fit_transform(train["email_address"].astype(str))
#%%
train["email_address_code"] # need to remove the "Total" datapoint though
#%%
train.drop("TOTAL", axis=0, inplace=True)
#%%
#total number of data points: train.shape[0] = 145
train.shape[0]
#%%
#allocation across classes (POI/non-POI): 127 non-pois and only 18 pois
train["poi"].value_counts()
#%%
#number of features used --> 20 = train.drop(target,axis=1).shape[1]-1
train.drop(target,axis=1).shape[1]-1
#%%
# need to take a closer look at the data
train.describe() # if there is no -1 in the data we can impute -1 for missing values. Should not impute 0, we have some 0's
# we have some huge stds, we can visualize outliers better later
# we have a few columns with almost all NaN. 
# They might hold some information, we can't be sure.
#%% # test for -1 in data
for col in train.drop(["email_address", "email_address_code"],axis=1).columns:
    if np.isin(-1, train[col].astype(np.float)):
        print("-1 is in column ", col,"!")
    else:
        print("nope for col", col)
# no -1 in data beside the -1 we put in email_address fields--> we can impute -1 for all nans
#%%
train.fillna(value = -1, inplace = True)
train.info()    
#%%
train.sample(5)
#%%
# now we drop the non encoded email column 
train.drop("email_address", axis=1, inplace=True)
train.info()
# ok we have a cleaned dataframe with some imputations and label encodings
# now we can apply scaling and do some automated feature engineering
# with interaction variables

#%%
data1 = train.iloc[:,:]
#%%
predictors = data1.drop(target, axis=1).columns
#%%
### Task 2: Remove outliers # use outlier removal techniques and check validation error with kfold
# First, we run through the the code without removing any data and give our clf results a good look
# Then, we test the effect of removing datapoints that have outliers in more than a set number of categories and test the impact on clf results
    


# =============================================================================
# now we remove outliers and take a look at classification results
# =============================================================================
"""I am using als.print_outliers() for this. 
This function looks for outliers in each of the features, 
that are lying __how_far__ steps away from its respective interquartile range. 
Each of these bad points are counted in a dictionary and finally bad points to be discarded are selected 
as those that occurred with highest frequency i.e., points that were bad in most features 
(as determined by the __worst_th__ parameter).
"""
from collections import defaultdict
# For each feature print the rows which have outliers in worst_th features 
def als_print_outliers(data, how_far=2, worst_th=6, to_display=False):
    #data = data.iloc[:,11:30] # use all the data for this
    really_bad_data = defaultdict(int)  # initiate dict for counting instances 
    for col in data.columns: # cycle through all the provided columns
        Q1 = np.percentile(data[col], 25) # set the Q1 
        Q3 = np.percentile(data[col], 75) # set the Q3 to get at the interquartile range
        step = (Q3-Q1)*how_far # how far away from the interquartile range do we allow outliers to be?
        bad_data = list(data[~((data[col]>=Q1-step)&(data[col]<=Q3+step))].index)  
        # use ~ operator to get the data indices that are not in the ranges as list
        for i in bad_data:
            really_bad_data[i]+= 1 # sum up the instances of bad data in any column at this index
    # Display the outliers
    max_ind = max(really_bad_data.values()) # how many bad data occurences do we have at maximum at any index location
    worst_points = [k for k, v in really_bad_data.items() if v > max_ind-worst_th] # get the index out of the dict only if the 
    # number of instances for that index is > (max bad data occurences for any index) ("max bad data occured in that one index for whole 
    #dataset") 
    # - threshhold for kicking out an index when it has at least that many bad data instances))
    if to_display:
        print("Data points considered outliers are:") 
        display(data.loc[worst_points,:]) # when using jupyter
    return worst_points



outlier_indices = als_print_outliers(data1[predictors], how_far = 2, worst_th=4, to_display = True)
#%%
# we can remove those datapoints but... we will remove some POIs with them. 
print("datapoints to remove:", outlier_indices)
# Let's test the effect on our scores... dropping just 2 outliers improved our scores so we stick with that
# Cleaning dataset by dropping outliers (cl)
data1_cl = data1.drop(outlier_indices,axis=0) # cleaned data1
#%%
data1_cl.info()
#%%
#... and we take a look at the data and different plots, focus is on pois
#g=sns.pairplot(data1_cl, hue=target)
#g.fig.set_size_inches(20,20)
# still have some outliers, but removing them hurts f1 score 
# so we abstain from doing that. We can see that POI can be separated according to a few columns like 
# director fees,  some stock columns and maybe some email features
#%% 
"""
#test of smote at beginning of data workflow before creation of interaction terms

# =============================================================================
# test effect of oversampling and SMOTE as well !
# in a new project we should probably start oversampling at the beginning - 
# not after creating interaction terms and weeding out collinear variables?
# =============================================================================

data = data1_cl

oversampling_data = data.copy(deep=True)

for i in range(0,1):
    oversampling_data = pd.concat([oversampling_data,oversampling_data[oversampling_data["poi"] == 1]],axis=0)
oversampling_data["poi"].value_counts()  

oversampling_data.info()
# now we have a rather balanced dataset, lets look at the impact of using the same data multiple times...
# results for different numbers of POIs below
#%%
data1_cl.index

#%%
from imblearn import over_sampling
#data = scaled_data
data = data1_cl
predictors = data.drop(target,axis=1).columns
smote_item = over_sampling.BorderlineSMOTE()
X_re, y_re = smote_item.fit_resample(data[predictors],data[target])
X_re = pd.DataFrame(X_re)
y_re = pd.DataFrame(y_re)

X_re.columns=predictors
y_re.columns=[target]
oversampling_blsmote = pd.concat([X_re, y_re],axis=1)
oversampling_blsmote.sample(5)
#%%
oversampling_blsmote.info()
#%%
oversampling_blsmote["poi"].value_counts()
"""
#%%
data = data1_cl
#data1 = train
#data = oversampling_blsmote
predictors = data.drop(target,axis=1).columns
index = data.index # need to conserve index - dont need index for smote

scaler = MinMaxScaler(feature_range=(0.0001,1))

scaler.fit(data[predictors])
scaled_data = scaler.transform(data[predictors])
scaled_data = pd.DataFrame(scaled_data, columns = predictors, index = index) # no need for smote
#scaled_data = pd.DataFrame(scaled_data, columns = predictors)
scaled_data = pd.concat([scaled_data, data[target]], axis=1)
#%%
scaled_data.sample(10) # works 
# now we can apply some automated feature engineering
#%%
scaled_data.info()
#%%

### Task 3: Create new feature(s) # apply this on scaled features and do it automatically. 
# Why? We do dozens of combinations and we want the importance of each feature to be comparable to all of the other features individually. 
# For example, we think about combining features like "Number of total emails written by person" and "Different unique persons our person wrote to", 
# the former feature value will be much higher than number of different persons that person wrote to, 
# so we need to scale or otherwise lose the informative value of number of different persons when combining those two features
# we will use automatic feature creation since we are lazy and remove redundant and correlated features with feature selection mechanics.


# create interaction terms for classification

def create_interaction_variables(data, columns):
    numerics = data.loc[:, columns] # apply this only on numeric columns without target column
    # for each pair of variables, determine which mathmatical operators to use
    for i in range(0, numerics.columns.size-1):
        for j in range(0, numerics.columns.size-1):
            col1 = str(numerics.columns.values[i])
            col2 = str(numerics.columns.values[j])
            # multiply fields together (we allow values to be squared)
            if i <= j:
                name = col1 + "*" + col2
                data = pd.concat([data, pd.Series(numerics.iloc[:,i] * numerics.iloc[:,j], name = name)], axis = 1)
            # add fields together
            if i < j:
                name = col1 + "+" + col2
                data = pd.concat([data, pd.Series(numerics.iloc[:,i] + numerics.iloc[:,j], name=name)], axis = 1)
            # divide and subtract fields from each other
            if not i == j:
                name = col1 + "/" + col2
                data = pd.concat([data, pd.Series(numerics.iloc[:,i] / numerics.iloc[:,j], name = name)], axis = 1)
                name = col1 + "-" + col2
                data = pd.concat([data, pd.Series(numerics.iloc[:,i] - numerics.iloc[:,j], name= name)], axis = 1)
        print("Column {} done, moving on to next column.".format(col1))
    return data

predictors = scaled_data.drop(target,axis=1).columns
scaled_data = create_interaction_variables(scaled_data, predictors)
#%%
scaled_data.info()
#1066 columns, so created roughly 1045
#%%
scaled_data.sample(10)
#%%
# opt to drop the columns with nan or inf values
def del_nan_or_inf_cols(data):
    for col in data.columns:
        if np.all(~np.isnan(data[col])) and np.all(np.isfinite(data[col])):
            continue
        else:
            data.drop([col], axis=1, inplace=True)
            print("dropped col: ",col)
#    return data

del_nan_or_inf_cols(scaled_data)
#%%
def remove_correlated_variables(data, columns, target, inplace = False):
    # calculate the correlation matrix for the predictors only
    df_corr = data.corr(method='spearman')
    # create a mask to ignore self-correlation
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask * df_corr
    
    drops = []
    # loop through each variable, drop the target from analysis though
    for col in df_corr.drop(target,axis=1).columns.values:
        # if we've already determined to drop the current variable, continue
        if np.isin([col], drops):
            continue
        # find all the variables that are highly correlated with the current variable
        # and add them to the drop list
        corr = df_corr[abs(df_corr[col]) > 0.93].index #remove all above 0.9x correlation-we varied x for tests
        drops = np.union1d(drops, corr)
    
    print("\nDropping", drops.shape[0], "highly correlated features...n", drops)
    return data.drop(drops, axis=1, inplace=inplace)

remove_correlated_variables(scaled_data, predictors, target, inplace = True)
#%%
scaled_data.info()

#%%
"""
# =============================================================================
# adding original variables and target to the dataset
# =============================================================================
# do we have all the original predictors in the scaled_data?
np.isin(train.columns,scaled_data.columns) 
#if not...
#%%
#data = scaled_data_vif
data = scaled_data
predictors = data.drop(target, axis=1).columns.values
# add all the original predictors to the dataset and let XGB find the best predictor set for best CV score
# by using the feature importances of XGB
for x in train.drop(target, axis=1).columns:
    if x not in predictors:
        data[x] = train[x]
# reassigning predictors
predictors = data.drop(target, axis=1).columns.values
data.info()
#%%

scaled_data.info()
"""
#%%
# =============================================================================
# # test the variables for effect on classification
 # splitting strategy
# =============================================================================
#cv_split = StratifiedShuffleSplit(n_splits = 4, random_state = 42)

cv_split = StratifiedShuffleSplit(n_splits = 10, random_state = 42) # more splits for oversampled data

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# we compare different clfs, application of pca and feature selection techniques via cv (maybe pipelined)


#%%
# Comparison of some models

# Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
       ('rfr', ensemble.RandomForestClassifier(n_jobs = 8, random_state=42)),
       ('lr', linear_model.LogisticRegression()),
       ('knn', neighbors.KNeighborsClassifier()),
       ('svc', svm.SVC()),
       ('nusvc', svm.NuSVC(nu=0.1)), # we have a little above 10% positive cases (POI) in the dataset
       ('linearsvc', svm.LinearSVC()),       
       ('xgb', xgboost.XGBClassifier(random_state=42, n_jobs=8)),
       ('NB', GaussianNB())
       ]

#%%
# =============================================================================
# apply scaling to data for some MLA algorithms. 
# Ideally do this in a cv pipeline with transformations on train and test
# =============================================================================

#unclean data
#data = train
#test with cleaned data (outliers removed)
#data = data1_cl
datatarget = data1_cl
#datatarget = oversampling_blsmote

#test with interaction data without outliers
#scaled_data = pd.concat([scaled_data, datatarget[target]], axis=1)
data = scaled_data

index = data.index
columns = data.drop(target,axis=1).columns

scaler = StandardScaler()

scaler.fit(data[columns])
scaled_data = scaler.transform(data[columns])
scaled_data = pd.DataFrame(scaled_data, columns = columns, index = index) # no need for smote
#scaled_data = pd.DataFrame(scaled_data, columns = columns)
scaled_data = pd.concat([scaled_data, datatarget[target]], axis=1)
#%%
scaled_data.sample(10)
# 300 !

#%% 

# =============================================================================
# test effect of oversampling and SMOTE as well !
# in a new project we should probably start oversampling at the beginning - 
# not after creating interaction terms and weeding out collinear variables?
# =============================================================================

oversampling_data = scaled_data.copy(deep=True)

for i in range(0,1):
    oversampling_data = pd.concat([oversampling_data,oversampling_data[oversampling_data["poi"] == 1]],axis=0)
oversampling_data["poi"].value_counts()  

oversampling_data.info()
# now we have a rather balanced dataset, lets look at the impact of using the same data multiple times...
# results for different numbers of POIs below
#%%
# =============================================================================
# # oversampling with smote after feature engineering
# =============================================================================
from imblearn import over_sampling
data = scaled_data
predictors = data.drop(target,axis=1).columns
smote_item = over_sampling.BorderlineSMOTE(random_state = 42)
X_re, y_re = smote_item.fit_resample(scaled_data[predictors],scaled_data[target])

X_re = pd.DataFrame(X_re)
y_re = pd.DataFrame(y_re)
X_re.columns=predictors
y_re.columns=[target]
oversampling_blsmote = pd.concat([X_re, y_re],axis=1)
oversampling_blsmote.sample(5)
#%%
oversampling_blsmote.info()
#%%
oversampling_blsmote["poi"].value_counts()
# even dataset now - 50:50!

#%%

#data = scaled_data
#data = oversampling_data
data = oversampling_blsmote
#data = scaled_data_vif
#data = scaled_data1
predictors=data.drop(target,axis=1).columns
#predictors = xgb_preds
#predictors = xgb_uncleaned_data_pred
#data = data1_dummy
#predictors = data1_dummy.drop(target, axis=1).columns

#create table to compare MLA metrics
MLA_columns = ["MLA Name", "MLA Train  f1 Mean", "MLA Test  f1 Mean", "MLA Test  f1 3*STD", "MLA TIME"]
MLA_compare = pd.DataFrame(columns=MLA_columns)
# index through MLA and save performance to table
row_index = 0
for name, alg in MLA:
    #set name and params
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, "MLA Name"] = MLA_name
  
    #score model with cross validation
    #we score with f1 score for the best classifier
    cv_results = model_selection.cross_validate(alg, data[predictors], data[target], cv=cv_split, 
                                                scoring = "f1", return_train_score=True, n_jobs = -1, verbose = True)

    MLA_compare.loc[row_index, "MLA TIME"] = cv_results["fit_time"].mean()
    MLA_compare.loc[row_index, "MLA Train  f1 Mean"] = cv_results["train_score"].mean()
    MLA_compare.loc[row_index, "MLA Test  f1 Mean"] = cv_results["test_score"].mean()
    #if this is a non-bias random sample, then +/-3 std from the mean, should statistically capture 99,7% of the subsets
    MLA_compare.loc[row_index, "MLA Test  f1 3*STD"] = cv_results["test_score"].std()*3
    #let's know the worst that can happen
    
    row_index+=1
# print and sort table:
MLA_compare.sort_values(by= ["MLA Test  f1 Mean"], ascending = False, inplace=True)
MLA_compare
 #%%
"""
results for original predictor set with 4 folds:
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
0  RandomForestClassifier           0.937224                 0   
1      LogisticRegression           0.584416                 0   
2    KNeighborsClassifier           0.185415                 0   
3                     SVC           0.571429                 0   
4                   NuSVC           0.965517                 0   
5               LinearSVC           0.674783                 0   
6           XGBClassifier                  1                 0   

  MLA Test  f1 3*STD     MLA TIME  
0                  0     0.109176  
1                  0    0.0043052  
2                  0  0.000978072  
3                  0   0.00166225  
4                  0   0.00132322  
5                  0   0.00430886  
6                  0    0.0495253 

#results for scaled_data without outliers and with interaction terms - 4folds and no original predictors with 4 folds
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
1      LogisticRegression           0.965517              0.45   
5               LinearSVC                  1          0.283333   
6           XGBClassifier                  1             0.125   
0  RandomForestClassifier           0.964286                 0   
2    KNeighborsClassifier           0.173407                 0   
3                     SVC           0.551948                 0   
4                   NuSVC           0.956281                 0   

  MLA Test  f1 3*STD    MLA TIME  
1               0.15  0.00921851  
5           0.497494   0.0196902  
6           0.649519   0.0668142  
0                  0    0.108979  
2                  0  0.00148851  
3                  0   0.0032351  
4                  0  0.00372791  


# results for scaled_data without outliers and with interaction terms - 4folds and with original predictors in the dataset with 4 folds
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
1      LogisticRegression           0.965517          0.516667   
5               LinearSVC                  1              0.35   
6           XGBClassifier                  1             0.125   
0  RandomForestClassifier           0.954365                 0   
2    KNeighborsClassifier           0.173407                 0   
3                     SVC           0.569805                 0   
4                   NuSVC           0.956281                 0   

  MLA Test  f1 3*STD    MLA TIME  
1           0.287228  0.00972116  
5            0.71239   0.0206946  
6           0.649519   0.0692934  
0                  0    0.109711  
2                  0  0.00149012  
3                  0  0.00349039  
4                  0  0.00373471  
    
"""
#results for VIF removed data without outliers with 4 folds
"""
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
6           XGBClassifier           0.973522          0.166667   
2    KNeighborsClassifier           0.304855             0.125   
4                   NuSVC           0.212023          0.104167   
0  RandomForestClassifier           0.907283                 0   
1      LogisticRegression                  0                 0   
3                     SVC                  0                 0   
5               LinearSVC                  0                 0   

  MLA Test  f1 3*STD    MLA TIME  
6           0.866025   0.0506229  
2           0.649519  0.00123513  
4            0.32476  0.00098598  
0                  0    0.110603  
1                  0  0.00348383  
3                  0  0.00149488  
5                  0  0.00298917 
"""

#all with with 4 folds
#results for dataset with all the interaction terms - 1065 variables r 0.98
"""
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
1      LogisticRegression                  1          0.436111   
5               LinearSVC                  1          0.425397   
4                   NuSVC           0.947044          0.416667   
0  RandomForestClassifier           0.943681          0.166667   
2    KNeighborsClassifier            0.25215                 0   
3                     SVC           0.569805                 0   
6           XGBClassifier                  1                 0   

  MLA Test  f1 3*STD    MLA TIME  
1           0.123322   0.0406901  
5           0.306838    0.141615  
4               0.75   0.0122014  
0           0.866025    0.110868  
2                  0  0.00447643  
3                  0   0.0124632  
6                  0     0.13189  

"""

#results for dataset with all the interaction terms - 363 variables r 0.95
"""
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
1      LogisticRegression                  1          0.502778   
5               LinearSVC           0.991379          0.392857   
0  RandomForestClassifier           0.954365               0.1   
2    KNeighborsClassifier          0.0919118                 0   
3                     SVC           0.569805                 0   
4                   NuSVC           0.956281                 0   
6           XGBClassifier                  1                 0   

  MLA Test  f1 3*STD    MLA TIME  
1           0.303109   0.0137135  
5           0.771792   0.0383708  
0           0.519615    0.109387  
2                  0  0.00197983  
3                  0  0.00473243  
4                  0  0.00498658  
6                  0   0.0805084  

"""

#results for dataset with all the interaction terms - 299 variables r 0.93 without original predictors 
# we take that dataset for further work!
"""
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
1      LogisticRegression           0.982759          0.553968   
5               LinearSVC                  1          0.455357   
4                   NuSVC           0.956281          0.166667   
0  RandomForestClassifier           0.954981                 0   
2    KNeighborsClassifier           0.204657                 0   
3                     SVC           0.569805                 0   
6           XGBClassifier                  1                 0   

  MLA Test  f1 3*STD    MLA TIME  
1           0.466059    0.011455  
5           0.509085   0.0316651  
4           0.866025  0.00398451  
0                  0    0.109635  
2                  0  0.00173992  
3                  0  0.00399244  
6                  0   0.0735254  

""" 
#results for dataset with all the interaction terms - 232 variables r 0.91 without original predictors 
"""
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
1      LogisticRegression           0.974138          0.516667   
5               LinearSVC                  1           0.47619   
6           XGBClassifier                  1              0.25   
4                   NuSVC           0.956281          0.166667   
0  RandomForestClassifier           0.946429                 0   
2    KNeighborsClassifier            0.09375                 0   
3                     SVC           0.569805                 0   

  MLA Test  f1 3*STD    MLA TIME  
1           0.287228  0.00996608  
5           0.571429   0.0234245  
6            1.29904   0.0681071  
4           0.866025  0.00373715  
0                  0    0.109815  
2                  0  0.00173944  
3                  0  0.00348699  

""" 

#results for dataset with all the interaction terms - 148 variables r 0.85 - with original predictors 
"""
5               LinearSVC                  1          0.421429   
1      LogisticRegression           0.928571             0.375   
6           XGBClassifier                  1               0.2   
0  RandomForestClassifier           0.964286                 0   
2    KNeighborsClassifier             0.0625                 0   
3                     SVC           0.587662                 0   
4                   NuSVC           0.964901                 0   

  MLA Test  f1 3*STD    MLA TIME  
5           0.265057    0.010716  
1           0.649519  0.00673032  
6            1.03923   0.0605761  
0                  0    0.108609  
2                  0  0.00174034  
3                  0  0.00223792  
4                  0  0.00298643  
""" 
#results for scaled data with interaction terms and removal of r > 0.9 columns
"""
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
1      LogisticRegression           0.972414          0.563333   
6           XGBClassifier                  1          0.533333   
5               LinearSVC                  1          0.417143   
4                   NuSVC           0.972414               0.1   
0  RandomForestClassifier           0.931429                 0   
2    KNeighborsClassifier           0.185217                 0   
3                     SVC           0.527068                 0   

  MLA Test  f1 3*STD    MLA TIME  
1           0.770973   0.0311997  
6                0.8    0.277678  
5           0.238464   0.0405597  
4                0.6     0.01248  
0                  0      0.0156  
2                  0  0.00623999  
3                  0  0.00311995 
""" 
# logistic regression is a very good choice!
# results for cv_split with only 3 folds:
"""
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
1      LogisticRegression           0.965517          0.722222   
5               LinearSVC                  1          0.466667   
6           XGBClassifier                  1          0.444444   
0  RandomForestClassifier           0.909524                 0   
2    KNeighborsClassifier              0.125                 0   
3                     SVC           0.497494                 0   
4                   NuSVC           0.965517                 0   

  MLA Test  f1 3*STD    MLA TIME  
1            0.62361   0.0208001  
5           0.141421   0.0259999  
6           0.942809      0.2548  
0                  0   0.0156002  
2                  0  0.00519999  
3                  0   0.0103999  
4                  0  0.00519999 
"""
# 4 folds
"""
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
1      LogisticRegression           0.965517          0.641667   
6           XGBClassifier                  1               0.5   
5               LinearSVC                  1              0.45   
4                   NuSVC           0.965517             0.125   
0  RandomForestClassifier           0.932143                 0   
2    KNeighborsClassifier           0.172697                 0   
3                     SVC           0.515977                 0   

  MLA Test  f1 3*STD    MLA TIME  
1           0.683283      0.4524  
6           0.866025      0.2613  
5               0.15       0.039  
4           0.649519  0.00389999  
0                  0   0.0156001  
2                  0  0.00390005  
3                  0  0.00389999  
"""

# =============================================================================
# # 4folds gaussian nb included - r 0.93 dataset
# =============================================================================
"""
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
1      LogisticRegression           0.982759          0.553968   
5               LinearSVC           0.991379          0.455357   
7              GaussianNB           0.441611          0.424242   
4                   NuSVC           0.956281          0.166667   
0  RandomForestClassifier           0.945744                 0   
2    KNeighborsClassifier            0.23223                 0   
3                     SVC           0.569805                 0   
6           XGBClassifier                  1                 0   

  MLA Test  f1 3*STD    MLA TIME  
1           0.466059   0.0119544  
5           0.509085   0.0326501  
7           0.229534  0.00199568  
4           0.866025  0.00396299  
0                  0    0.109707  
2                  0  0.00174534  
3                  0  0.00398326  
6                  0   0.0770213  
"""
# =============================================================================
# #results for oversampling (136 pois in dataset)
# this improves our f1 score for the train data... but we need to predict better on
# test data we haven't seen yet
# try using less POIs and maybe improve model hyperparameter tuning that way
# rfc untuned f1 score on tester.py is 0.234
# =============================================================================
"""
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
0  RandomForestClassifier            0.99898          0.983333   
4                   NuSVC                  1          0.966667   
6           XGBClassifier                  1          0.966092   
5               LinearSVC                  1          0.958046   
1      LogisticRegression                  1          0.926796   
3                     SVC           0.962429          0.919221   
2    KNeighborsClassifier           0.910479          0.910753   
7              GaussianNB           0.861448          0.876716   

  MLA Test  f1 3*STD    MLA TIME  
0          0.0866025      0.1326  
4                0.1      0.0117  
6          0.0707317     0.62815  
5          0.0826688      0.1248  
1          0.0979907   0.0351002  
3           0.212572   0.0155999  
2          0.0391108           0  
7           0.116578  0.00779998 
"""


# =============================================================================
# #results for oversampling (34 pois in dataset)
# tune model hyperparameters with this dataset now
# use LinearSVC # results worse on tester than with normal dataset.
# try a run with 136 pois
# =============================================================================
"""
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
0  RandomForestClassifier           0.996032          0.964286   
6           XGBClassifier                  1          0.892857   
4                   NuSVC                  1          0.839286   
5               LinearSVC                  1          0.839286   
1      LogisticRegression           0.995902          0.758929   
7              GaussianNB            0.62022          0.581006   
2    KNeighborsClassifier           0.541447             0.375   
3                     SVC           0.746043               0.2   

  MLA Test  f1 3*STD    MLA TIME  
0           0.185577    0.124838  
6           0.185577    0.527321  
4           0.307744   0.0117072  
5           0.307744   0.0702427  
1           0.350269   0.0273166  
7           0.346691  0.00390238  
2            0.73951  0.00390238  
3            1.03923   0.0117071  
"""
# 136 POIs
"""
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
0  RandomForestClassifier            0.99898          0.983333   
4                   NuSVC                  1          0.966667   
6           XGBClassifier                  1          0.966092   
5               LinearSVC                  1          0.958046   
1      LogisticRegression                  1          0.926796   
3                     SVC           0.962429          0.919221   
2    KNeighborsClassifier           0.910479          0.910753   
7              GaussianNB           0.861448          0.876716   

  MLA Test  f1 3*STD    MLA TIME  
0          0.0866025    0.128702  
4                0.1   0.0156002  
6          0.0707317    0.405605  
5          0.0826688    0.109201  
1          0.0979907   0.0390005  
3           0.212572   0.0156001  
2          0.0391108  0.00389999  
7           0.116578  0.00390011 
"""
# =============================================================================
# #results for borderline SMOTE oversampling_blsmote
# (126 pois in dataset now - 50:50 distribution of classes)
# tester py already has AMAZING results for that dataset without tuning. 
# Could not believe it!
# tune model hyperparameters with this dataset now - use SVC?
# 10 folds used here!
# =============================================================================
"""
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
6           XGBClassifier                  1          0.943127   
4                   NuSVC                  1          0.936577   
0  RandomForestClassifier           0.996893            0.9346   
3                     SVC           0.968006           0.92734   
5               LinearSVC                  1          0.919411   
1      LogisticRegression                  1          0.896671   
2    KNeighborsClassifier           0.887908          0.864865   
7              GaussianNB           0.866846          0.848756   

  MLA Test  f1 3*STD    MLA TIME  
6           0.094965     0.70512  
4          0.0698211  0.00780005  
0           0.138569     0.14664  
3           0.108976      0.0156  
5           0.157426     0.09516  
1           0.151522     0.03432  
2           0.138702  0.00312002  
7           0.138817  0.00880001  
"""
# results for NuSVC with 0.1
"""
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
4                   NuSVC           0.999119          0.944614   
6           XGBClassifier                  1          0.942658   
5               LinearSVC           0.999556          0.923685   
0  RandomForestClassifier           0.995548          0.918324   
3                     SVC           0.966849          0.913505   
1      LogisticRegression                  1          0.904105   
2    KNeighborsClassifier           0.901246          0.885276   
7              GaussianNB           0.869016          0.852581   

  MLA Test  f1 3*STD    MLA TIME  
4          0.0861644   0.0156003  
6           0.130685    0.842033  
5           0.143621    0.109201  
0           0.155564    0.138842  
3           0.144667   0.0171602  
1           0.136943   0.0296403  
2           0.143828  0.00312004  
7           0.116334           0  
"""

# =============================================================================
# #results for borderline SMOTE oversampling_blsmote
# oversampled at beginning of feature engineering workflow
# 10 folds used here!
# =============================================================================
"""
                 MLA Name MLA Train  f1 Mean MLA Test  f1 Mean  \
0  RandomForestClassifier           0.997774          0.961957   
6           XGBClassifier                  1          0.945741   
1      LogisticRegression                  1          0.935375   
3                     SVC           0.969651          0.924396   
5               LinearSVC                  1          0.922326   
4                   NuSVC           0.912541          0.900739   
2    KNeighborsClassifier           0.887449          0.876757   
7              GaussianNB           0.873925          0.843943   

  MLA Test  f1 3*STD    MLA TIME  
0           0.145038    0.134582  
6           0.107902    0.800285  
1           0.122056   0.0670804  
3           0.131037   0.0124801  
5           0.136886   0.0670804  
4           0.168296   0.0327602  
2           0.139796  0.00558014  
7            0.15964  0.00312002   
"""
#%%
# plot the results
sns.barplot(x="MLA Test  f1 Mean", y="MLA Name", data=MLA_compare, color="m")
plt.title("MLA f1 Score\n")
plt.xlabel("f1 Score")
plt.ylabel("Algorithm")

#%%
"""
def plot_pca(data, predictors, iterated_power, n_components):
    pca_item = PCA(iterated_power=iterated_power, n_components= n_components)
    components = pca_item.fit(data[predictors])
    transformed = pca_item.transform(data[predictors])
    
    var = components.explained_variance_ratio_ #amount of variance that each PC explains
    var1 = np.cumsum(np.round(components.explained_variance_ratio_,3)) #cumulative variance explained

    number_points = np.arange(1,n_components+1)
    plt.figure(figsize=(12,6))
    sns.set_style("whitegrid", {'axes.grid' : True})
    plt.plot(number_points, var1, marker="o", markerfacecolor="r", markersize=10)
    for x,y in zip(number_points,var1):
        plt.annotate(str(round(y,2)), xy=(x,y), xytext=(10,-10), textcoords = 'offset points')
    plt.show() 
    print("\nIndividual variance explained by the components: ", var)
    print("\nCumulative variance explained by adding each new component: ", var1)
    return pd.DataFrame(transformed)

predictors = scaled_data.drop(target, axis=1).columns
x_pca = plot_pca(scaled_data, predictors, 7, 100)
# do cv with PCA 5,10, 15, 20, 25, 30, 60, 70, 80, 100 components
"""
#%%

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
"""
grid_coeffs = [0, 0.1, 1, 10]
grid_gamma = [0.001, 0.01, 0.1, 1, 10, "auto"]
grid_degree = [2, 3, 4]
grid_bool = [True, False]
grid_seed = [42]
grid_regular = [1e-5,1e-4,1e-3]
grid_n_estimator = [5,10,15,30,50,70,90] 
grid_ratio = [0.03, .1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_coeffs = [0, 0.01, 0.03]
grid_gamma = [0.001, 0.01, 0.1, 1, "auto"]
grid_max_depth = [7,8,9,10,11,None]
#grid_max_depth = [3, 10, None]
grid_min_leaf_samples = [1, 5, 10, 30]
grid_max_leaf_nodes = [5, 10, 30]
#grid_min_samples = [5, 10, .03, .05, .10]
grid_min_samples = [2,3,4,5]
grid_criterion = ["gini", "entropy"]
#grid_features = [int(np.sqrt(len(predictors))),int(len(predictors)/2)+1, int(0.8*len(predictors)), len(predictors)-1]
grid_features = [1,2,3,4,5]

pcacomp = [x/100 for x in range(1,100)]
pcacomp.append(None)



grid_param_level0 = [
      [{
      #SVC
      "kernel": ["linear", "poly", "rbf", "sigmoid"],
      "C": [0.01, 0.1, 1, 10, 100, 1000], 
      "gamma": grid_gamma,
      "degree": grid_degree,
      "tol" : grid_regular,
      "decision_function_shape": ["ovo", "ovr"], 
      "coef0": grid_coeffs,
      'class_weight': ['balanced', None],
      "random_state": grid_seed,
      "verbose" :  [True]
      }],  

        [{ 
#      NuSVC
      "nu" : [0.01, 0.05, 0.1],
      "kernel": ["linear", "poly", "rbf", "sigmoid"],
      "coef0": grid_coeffs,
      "gamma": grid_gamma,
      "tol" : grid_regular,
      "decision_function_shape": ["ovo", "ovr"], 
      "degree" : grid_degree,
      "probability": [True],
      "random_state": grid_seed
      }], 
      
       [{ 
      #LinearSVC # 
      'penalty': ['l1','l2'], 
      'loss': ['squared_hinge','hinge'], 
      'dual': [False, True],
      'tol': grid_regular, 
      'C':  [0.01, 0.1, 1, 10, 100, 1000], 
      'fit_intercept': [True, False], 
      'intercept_scaling' : [0.3, 1, 3, 10], 
      'class_weight' : ['balanced', None], 
      'verbose' : [True],
      'random_state' :grid_seed, 
      'max_iter' : [1000],
      }], 
        ]
"""
    #%%
"""
# gridsearched here
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

#data = scaled_data
#data = oversampling_data # test with oversampled data (once, copied POIs = 36 pois) 
data = oversampling_blsmote # SMOTE!
#predictors = xgb_preds
predictors = data.drop(target, axis=1).columns

skbestk = [x for x in range(1,len(predictors))]
skbestk.append("all")

# we use randomizedsearchCV first and apply gridsearchCV after that in our pipeline, 
# cross validating different clfs and parameters

# Pipeline definition
cachedir = mkdtemp()
# Linear SVC is the best, PCA does not give any improvement 
# therefore we continue with SKBest and Linear SVC only
MLA_pipe = [

    [
#     ('pca', PCA()), # Might as well...
     ('feature_selection', SelectKBest()), # Let's be selective
     ('clf', svm.SVC())
     ], 
     
#    [
#     ('pca', PCA()), # Might as well...
#     ('feature_selection', SelectKBest()), # Let's be selective
#     ('clf', svm.NuSVC(nu=0.05))], #step 2....
    
#    [
#    ('pca', PCA()), # Might as well...
#    ('feature_selection', SelectKBest()), # Let's be selective
#    ('clf', svm.LinearSVC()) # Start simple
#    ],
    
#    [
#     ('pca', PCA()),
#     ('feature_selection', SelectKBest()), # Let's be selective
#     ('clf', linear_model.LogisticRegression())
#     ],
     
#     [
#      ('pca', PCA()),
#      ('feature_selection', SelectKBest()), # Let's be selective,
#      ('NB', GaussianNB())
#      ],
    
#    [
#     ('clf', ensemble.RandomForestClassifier()),
#     ],
]

#set up params to check
params = [ # xgb_preds is 12 in length
        [{ # too many parameters, have to tune stepwise
#        'pca__n_components': [0.99], # can select part of the features as components
        'feature_selection__k' : [x for x in range(35,52,2)],#[x for x in range(1,300,9)],#,90,100,110,120,130,140, 'all'], # can test effect of selecting all and only part of the features/components
        'clf__degree': [2,3], 
          'clf__kernel': ["poly", "rbf", "sigmoid"], # we cancel out "linear" here
#          "clf__decision_function_shape": ["ovo", "ovr"], 
          'clf__gamma' : [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 3],
#          'clf__tol': grid_regular, 
#          'clf__coef0': grid_coeffs,
          'clf__class_weight' : ['balanced'],#, None], 
          'clf__verbose' : [False],
          'clf__random_state' :grid_seed, 
          'clf__C' : [0.1, 1, 3, 10, 100, 1000, 10000],
          }],
        
#        [{
##        'pca__n_components': [None, .1, .2, .3, .4, .45, .5 ,.55, .6, .65, .7, .75, .8, .85, .9, .95, .99], 
#        'feature_selection__k' : [15,20,25,30,35],#['all',30,35,40,45,50,80,100,120,200], # can test effect of selecting all and only part of the features/components
#        'clf__degree': [2,3], 
#          'clf__gamma' : [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1],
#          'clf__kernel': ["poly", "rbf", "sigmoid"], # we cancel out "linear" here
##          'clf__tol': grid_regular, 
#          'clf__class_weight' : ['balanced'],#, None], # no None for us! 
#          'clf__verbose' : [True],
##          'clf__coef0': grid_coeffs,
#          'clf__random_state' :grid_seed, 
#            "clf__nu" : [0.5, 0.01, 0.03, 0.1, 0.2, 0.3],
#                }],
#        
#        {
##        'pca__n_components': [0.96], # can select part of the features as components
#        'feature_selection__k' : [50], # can test effect of selecting all and only part of the features/components
#        'clf__penalty': ['l2'], 
#          'clf__loss': ['squared_hinge'], # squared seems better than 'hinge'
#          'clf__tol': grid_regular, 
#          'clf__fit_intercept': [True], # True seems better
##          'clf__intercept_scaling' : [0.3, 1, 3, 10], 
#          'clf__class_weight' : ['balanced'],#, None], # no None for us! 
#          'clf__verbose' : [False],
#          'clf__random_state' :grid_seed, 
#          'clf__max_iter' : [1000],
#          'clf__C' : [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 30,100, 300, 1000, 3000, 10000, 30000],
#        },

#            [{
#        'pca__n_components': [0.98], # can select part of the features as components
#         'feature_selection__k' : [23], # can test effect of selecting all and only part of the features/components
#         'clf__solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag' , 'saga'], 
#         'clf__penalty': ['l2'], 
#          'clf__tol': [1e-4,1e-5,1e-3], 
#          'clf__fit_intercept': [True, False], 
#          'clf__class_weight' : ['balanced'],#, None], 
#          'clf__verbose' : [False],
#          'clf__random_state' :grid_seed, 
#          'clf__max_iter' : [1000],
#          'clf__C' : [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000,3000, 10000, 30000],
#                }],
#    [{
#        'pca__n_components': pcacomp, # can select part of the features as components
#        'feature_selection__k' : skbestk, # can test effect of selecting all and only part of the features/components
#                }],
    
    
#            {
#      #RandomForestClassifier
#      "clf__n_jobs": [8],
#      "clf__max_features" : grid_features, # 11 was good 
#      "clf__n_estimators": grid_n_estimator, 
#      "clf__max_depth": grid_max_depth,   
#      # max_depth should only be tuned without setting values for min_samples_leaf and max_leaf_nodes
#      # they both cancel out the effect of setting max_depth
#      "clf__oob_score": [True], #def = False
#      "clf__min_samples_split" : grid_min_samples, # 10 was good
#      "clf__criterion" : grid_criterion,
#      #"min_samples_leaf" : grid_min_leaf_samples,
#      #"max_leaf_nodes": grid_max_leaf_nodes, 
#      "clf__random_state": grid_seed
#      },
]

# set up dataframe for results
MLA_columns = ["MLA Name", "train f1", "test f1", "test f1 3*STD", "time"]
MLA_compare = pd.DataFrame(columns=MLA_columns)
row_index = 0
#set timer for total runtime 
start_total = time.perf_counter()

# put the whole thing in cv loop over train set....
best_estimators = []
best_parameters = []

for clf, param in zip(MLA_pipe, params):
    #set timer for cv runtime
    start = time.perf_counter()
    #MLA is a list of tuples, index 0 is the name and index 1 is the algorithm
    #grid_param is a list of param_grids for the gridsearch for each estimator
    # do param search
    print("started with ", clf[-1][1].__class__.__name__)
    train_f1_score=[]
    train_precision_score=[]
    train_recall_score=[]
    test_auc_score=[]
    test_f1_score=[]
    test_precision_score=[]
    test_recall_score=[]

    min_target = -1
    best_params = None
    
    #get the pipeline object
    pipeline = Pipeline(clf, memory = cachedir)   
# =============================================================================
#     gridsearching     
# =============================================================================
#    create gridsearch clf
    model = model_selection.GridSearchCV(pipeline, param_grid=param, cv = cv_split, iid=False, scoring = "f1", verbose = 10, return_train_score = True, n_jobs = 3)

    # Now fit model on the data
    model.fit(data[predictors], data[target])
# =============================================================================
#     randomizedsearch
# =============================================================================
#    model = RandomizedSearchCV(estimator = pipeline, param_distributions = param, n_iter = 1000, cv = cv_split, verbose = 2, random_state = 42, n_jobs = 2, return_train_score = True)
## best params - gridsearch over this range near that:
#    # fit the random search # covering n_iter combinations instead of every combination
#    model.fit(data[predictors], data[target])
#    
    #if fold is better on test set than another we use that as our best model and parameters
    mean_target = model.cv_results_["mean_test_score"][model.best_index_]
    if mean_target > min_target:
        min_target = mean_target
        best_algorithm = model.best_estimator_
        best_params = model.best_params_
        best_scores = model.cv_results_["mean_test_score"][model.best_index_]

    #store best parameters and estimators
    best_estimators.append(best_algorithm)
    best_parameters.append(best_params)
    
    # store results in DF
    MLA_compare.loc[row_index, "train f1"] = model.cv_results_["mean_train_score"][model.best_index_]
    MLA_compare.loc[row_index, "test f1"] = model.cv_results_["mean_test_score"][model.best_index_]
    MLA_compare.loc[row_index, "test f1 3*STD"] = model.cv_results_["std_test_score"][model.best_index_]*3
    MLA_compare.loc[row_index, "MLA Name"] = clf[-1][1].__class__.__name__
    duration = time.perf_counter() - start
    MLA_compare.loc[row_index, "time"] = "{:.0f}:{:.0f}:{:.1f}".format(\
      duration // 3600, (duration % 3600 // 60), duration % 60)
    row_index+=1

# print and sort table:
MLA_compare.sort_values(by= ["test f1"], ascending = False, inplace=True)
rmtree(cachedir)
# print total search runtime and best params 
endtotal = time.perf_counter() - start_total

print("\nBest params for best algorithm {}:  {}, f1-score: {}".format(best_algorithm, 
      best_params, min_target))
print('Total runtime is {:.0f}:{:.0f}:{:.0f}'.format(endtotal // 3600,
      (endtotal % 3600 // 60), endtotal % 60))
print(MLA_compare)
print("\n",best_estimators)
print("\n",best_parameters)
"""
 #%%
#results for non pca or skbest - full dataset with interaction variables! 
"""
    MLA Name  train f1   test f1 test f1 3*STD train prec test prec train rec  \
2  LinearSVC         1  0.616508      0.640759          1   0.50381         1   
0        SVC  0.778818       0.2      0.734847        0.8       0.2      0.76   
1      NuSVC  0.958128       0.1           0.6          1       0.1      0.92   

  test rec bestcount      time  
2      0.9         1  0:0:12.9  
0      0.2         1   0:0:5.0  
1      0.1         1   0:0:6.4  
"""

#results for skbest
"""
    MLA Name  train f1   test f1 test f1 3*STD train prec test prec train rec  \
2  LinearSVC  0.889843      0.29      0.891964   0.939706  0.233333  0.866667   
1      NuSVC  0.371145  0.238095      0.651529   0.352092      0.17  0.506667   
0        SVC  0.918291         0             0   0.968485         0      0.88   

  test rec bestcount      time  
2      0.4         1  0:1:17.1  
1      0.4         2  0:0:43.1  
0        0         0   0:1:5.7  
"""
#results for pca
"""
    MLA Name  train f1   test f1 test f1 3*STD train prec  test prec  \
2  LinearSVC  0.603054  0.506667       1.06283   0.567337       0.45   
1      NuSVC  0.933763      0.08          0.48          1  0.0666667   
0        SVC         0         0             0          0          0   

  train rec test rec      time  
2  0.653333      0.6  0:0:44.0  
1      0.88      0.1  0:0:31.7  
0         0        0  0:0:49.0 
"""
# therefore we don't apply PCA OR SKBest ... 
# continue tuning only LinearSVC

#added logreg - results for xgb_cleaned_interaction_pred
"""
             MLA Name  train f1   test f1 test f1 3*STD train prec test prec  \
3  LogisticRegression  0.414392  0.359365      0.654374    0.35844   0.26381   
2           LinearSVC  0.422823   0.33987      0.780929   0.339412  0.251111   
1               NuSVC  0.720676       0.1           0.6   0.802331       0.1   
0                 SVC  0.436842         0             0          1         0   

  train rec test rec      time  
3  0.653333      0.6  0:5:59.7  
2  0.746667      0.6  0:4:19.9  
1  0.666667      0.1   0:0:0.3  
0      0.28        0   0:0:0.2  
"""

#results for full dataset with removed variables r > 0.9
""" {'feature_selection__k': 15}, {'feature_selection__k': 15}
             MLA Name  train f1  test f1 test f1 3*STD train prec test prec  \
1  LogisticRegression  0.803184     0.53       0.52192   0.941667       0.6   
0           LinearSVC  0.804329  0.50381      0.447797   0.938095  0.606667   

  train rec test rec      time  
1  0.733333      0.6   0:0:3.3  
0  0.746667      0.5  0:0:10.8  
"""
# results for full dataset with removed variables r>0.9
""" no feature selection
             MLA Name  train f1   test f1 test f1 3*STD train prec test prec  \
1  LogisticRegression  0.972414  0.563333      0.770973          1       0.5   
0           LinearSVC         1  0.417143      0.238464          1  0.373333   

  train rec test rec     time  
1  0.946667      0.7  0:0:0.5  
0         1      0.5  0:0:1.3  
"""

# results for gridsearch full dataset with removed variables r>0.9
""" no feature selection - worse results!?!
             MLA Name  train f1   test f1 test f1 3*STD train prec test prec  \
0           LinearSVC  0.661882  0.489235      0.339363   0.569599  0.348254   
1  LogisticRegression  0.697762  0.466667      0.536656   0.608941  0.441667   

  train rec test rec      time  
0  0.986667      0.9  0:6:40.9  
1  0.946667      0.7   1:3:7.2  
"""
#results for gridsearch on full dataset without prediction on test folds, removed varibles r>0.9
"""
             MLA Name  train f1   test f1 test f1 3*STD       time
0           LinearSVC  0.495235  0.465397      0.729662   0:5:37.8
1  LogisticRegression  0.984073  0.457302      0.452704  0:25:40.0
steps=[('feature_selection', SelectKBest(k=100, score_func=<function f_classif at 0x000000000C577D08>)), ('clf', LogisticRegression(C=100, class_weight='balanced', dual=False,
          fit_intercept=False, intercept_scaling=1, max_iter=1000,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=42,
          solver='newton-cg', tol=0.0001, verbose=False, warm_start=False))])
"""
#results for gs full data with all variables
""" 
             MLA Name  train f1   test f1 test f1 3*STD      time
1  LogisticRegression  0.495077  0.433442      0.359308  0:7:33.0
0           LinearSVC  0.996552  0.416825      0.251861  0:0:58.8

 
     steps=[('clf', LinearSVC(C=1, class_weight='balanced', dual=True, fit_intercept=True,
     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
     penalty='l2', random_state=42, tol=0.0001, verbose=False))]), 
     steps=[('clf', LogisticRegression(C=0.001, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=1000,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=42,
          solver='saga', tol=0.001, verbose=False, warm_start=False)
"""
#results for gs full data with all variables-PCA 3folds
""" 
             MLA Name  train f1   test f1 test f1 3*STD     time
0           LinearSVC  0.697005  0.777778      0.471405  0:2:7.4
1  LogisticRegression         1   0.72381      0.323249  0:9:8.8


     steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=0.93, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', LinearSVC(C=1000, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,
     verbose=False))]), 
     steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=False,
          intercept_scaling=1, max_iter=1000, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=42, solver='newton-cg', tol=0.0001,
          verbose=False, warm_start=False))])]
"""

#results for gs full data with all variables-SKB 3folds
"""
             MLA Name  train f1   test f1 test f1 3*STD      time
0           LinearSVC  0.830769  0.822222      0.410961   0:1:5.9
1  LogisticRegression  0.558528  0.777778      0.471405  0:6:30.4

 
     steps=[('feature_selection', SelectKBest(k='all', score_func=<function f_classif at 0x000000000C577D08>)), 
     ('clf', LinearSVC(C=0.1, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
     penalty='l2', random_state=42, tol=0.0001, verbose=False))]),
     steps=[('feature_selection', SelectKBest(k=24, score_func=<function f_classif at 0x000000000C577D08>)), 
     ('clf', LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=1000, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=42, solver='liblinear', tol=0.0001,
          verbose=False, warm_start=False))])]
"""
# results for full data 3 folds no feature selection/reduction
"""
             MLA Name  train f1   test f1 test f1 3*STD      time
0           LinearSVC  0.830769  0.822222      0.410961  0:0:17.3
1  LogisticRegression  0.965517  0.722222       0.62361  0:2:19.5

 
     steps=[('clf', LinearSVC(C=0.1, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
     penalty='l2', random_state=42, tol=0.0001, verbose=False))]),
     steps=[('clf', LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=1000, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=42, solver='liblinear', tol=0.0001,
          verbose=False, warm_start=False))])]
"""


# results for full data r0.93 - 4 folds skbest # best score on tester py so far with LinearSVC!
"""
             MLA Name  train f1   test f1 test f1 3*STD       time
1  LogisticRegression  0.917967  0.733333           0.2  0:12:40.4
0           LinearSVC  0.831944  0.666667      0.612372    0:2:6.1


     steps=[('feature_selection', SelectKBest(k=100, score_func=<function f_classif at 0x00000260029019D8>)), 
     ('clf', LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
     penalty='l2', random_state=42, tol=0.0001, verbose=False))]), 
     steps=[('feature_selection', SelectKBest(k=200, score_func=<function f_classif at 0x00000260029019D8>)), 
     ('clf', LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=1000, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=42, solver='newton-cg', tol=0.0001,
          verbose=False, warm_start=False))])]

"""

# results for full data r0.93 - 4 folds PCA
"""
             MLA Name  train f1   test f1 test f1 3*STD      time
1  LogisticRegression         1  0.592857      0.369293  0:8:31.5
0           LinearSVC  0.483465  0.587302      0.758175  0:2:46.7

 
     steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=0.6, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', LinearSVC(C=0.1, class_weight='balanced', dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,
     verbose=False))]), 
     steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', LogisticRegression(C=10000, class_weight='balanced', dual=False,
          fit_intercept=False, intercept_scaling=1, max_iter=1000,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=42,
          solver='lbfgs', tol=0.001, verbose=False, warm_start=False))])]

"""
# results for full data r0.93 - 4 folds SKB with SVC tuning
"""
  MLA Name  train f1 test f1 test f1 3*STD     time
0      SVC  0.845299     0.7      0.173205  0:6:7.2

     steps=[('feature_selection', SelectKBest(k=90, score_func=<function f_classif at 0x00000260029019D8>)), 
     ('clf', SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=42, shrinking=True,
  tol=0.001, verbose=False)
"""

# results for full data r0.93 - 4 folds SKB with SVC tuning for oversampled POIs (36)
"""
    MLA Name  train f1 test f1 test f1 3*STD      time
0  LinearSVC  0.945543       1             0  0:6:13.2

     steps=[('feature_selection', SelectKBest(k=130, score_func=<function f_classif at 0x000000000C9986A8>)), 
     ('clf', LinearSVC(C=1, class_weight='balanced', dual=True, fit_intercept=True,
     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
     penalty='l2', random_state=42, tol=0.001, verbose=False))])]
"""

# results for full data r0.93 - 4 folds SKB with RFC tuning for oversampled POIs (36)
"""
                 MLA Name train f1 test f1 test f1 3*STD      time
0  RandomForestClassifier        1       1             0  0:1:28.0

     steps=[('clf', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=12, max_features=1, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=4,
            min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=1,
            oob_score=True, random_state=42, verbose=0, warm_start=False))])]
"""

""" svc skbest
    MLA Name  train f1   test f1 test f1 3*STD      time
0  LinearSVC  0.991746  0.878571      0.267643  0:5:24.1

     steps=[('feature_selection', SelectKBest(k=100, score_func=<function f_classif at 0x000000000C9986A8>)), ('clf', LinearSVC(C=10, class_weight='balanced', dual=True, fit_intercept=True,
     intercept_scaling=3, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,
     verbose=False))])]
    """

""" logreg skbest
             MLA Name train f1   test f1 test f1 3*STD       time
0  LogisticRegression        1  0.888571      0.352264  0:36:17.7

     steps=[('feature_selection', SelectKBest(k=90, score_func=<function f_classif at 0x000000000C9986A8>)), 
     ('clf', LogisticRegression(C=10000, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=1000,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=42,
          solver='newton-cg', tol=0.0001, verbose=False, warm_start=False))])]
"""
#results for oversampling smote
"""
                 MLA Name  train f1   test f1 test f1 3*STD      time
0  RandomForestClassifier  0.991119  0.984615     0.0923077  0:0:13.3

     steps=[('clf', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=8, max_features=2, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=6,
            min_weight_fraction_leaf=0.0, n_estimators=8, n_jobs=1,
            oob_score=True, random_state=42, verbose=0, warm_start=False))])]
"""
"""
    MLA Name  train f1   test f1 test f1 3*STD      time
0  LinearSVC  0.991135  0.978307     0.0861248  0:1:44.5

     steps=[('feature_selection', SelectKBest(k=100, score_func=<function f_classif at 0x000000000C647048>)), 
     ('clf', LinearSVC(C=1, class_weight='balanced', dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,
     verbose=False))])]
"""
#results for skbest 40 svc with smote - best so far on tester.py!
"""
  MLA Name train f1   test f1 test f1 3*STD       time
0      SVC        1  0.976889      0.056715  0:51:55.6

     steps=[('feature_selection', SelectKBest(k=40, score_func=<function f_classif at 0x000000000C647048>)), 
     ('clf', SVC(C=3, cache_size=200, class_weight='balanced', coef0=0,
  decision_function_shape='ovr', degree=2, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=42, shrinking=True,
  tol=0.0001, verbose=False))])]
"""
#results for skbest [35,52] svc with smote
"""
MLA Name train f1   test f1 test f1 3*STD      time
0      SVC        1  0.965757     0.0784438  0:3:46.3

     steps=[('feature_selection', SelectKBest(k=35, score_func=<function f_classif at 0x000000000C487D90>)),
     ('clf', SVC(C=10, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=2, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=42, shrinking=True,
  tol=0.001, verbose=False)
"""

#results for skbest svc with smote
"""
  MLA Name  train f1   test f1 test f1 3*STD      time
0      SVC  0.996452  0.967666       0.10991  0:12:8.5

     steps=[('feature_selection', SelectKBest(k=298, score_func=<function f_classif at 0x000000000C487D90>)), 
     ('clf', SVC(C=1000, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=2, gamma=0.0001, kernel='sigmoid',
  max_iter=-1, probability=False, random_state=42, shrinking=True,
  tol=0.001, verbose=False)
"""

#results for full dataset svc with smote
"""
  MLA Name train f1   test f1 test f1 3*STD      time
0      SVC        1  0.949159     0.0839628  0:5:46.2

     steps=[('clf', SVC(C=100, cache_size=200, class_weight='balanced', coef0=0.03,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='poly',
  max_iter=-1, probability=False, random_state=42, shrinking=True,
  tol=1e-05, verbose=False)

"""

#results for skbest logreg with smote
"""
             MLA Name train f1   test f1 test f1 3*STD       time
0  LogisticRegression        1  0.952988      0.126647  0:46:28.5

     steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('feature_selection', 
    SelectKBest(k=80, score_func=<function f_classif at 0x000000000C647048>)), 
    ('clf', LogisticRegression(C=100, class_weight='balanced', dual=False, penalty='l2', random_state=42,
          solver='newton-cg', tol=0.0001, verbose=False, warm_start=False))])]
"""

#results for skbest gaussianNB
"""
     MLA Name  train f1   test f1 test f1 3*STD      time
0  GaussianNB  0.895203  0.867067      0.187616  0:1:24.7

     steps=[('feature_selection', SelectKBest(k=101, score_func=<function f_classif at 0x000000000C487D90>)), 
     ('NB', GaussianNB(priors=None))])]
"""

#results for PCA with gaussianNB
"""
     MLA Name  train f1  test f1 test f1 3*STD      time
0  GaussianNB  0.967542  0.94558      0.131984  0:1:35.1

     steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('NB', GaussianNB(priors=None)
"""

#results for skb with nusvc
"""
  MLA Name  train f1   test f1 test f1 3*STD      time
0    NuSVC  0.989837  0.973757     0.0873405  0:3:37.1

     steps=[('feature_selection', SelectKBest(k=30, score_func=<function f_classif at 0x000000000C822E18>)), 
     ('clf', NuSVC(cache_size=200, class_weight='balanced', coef0=0.0,
   decision_function_shape='ovr', degree=2, gamma=0.001, kernel='rbf
   ,',
   max_iter=-1, nu=0.1, probability=False, random_state=42, shrinking=True,
   tol=0.001, verbose=True)
"""

#results for svc with pca
"""  
MLA Name  train f1   test f1 test f1 3*STD      time
0      SVC  0.997786  0.949617      0.104641  0:5:17.8

     steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=0.99, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=1000, cache_size=200, class_weight='balanced', coef0=0,
  decision_function_shape='ovr', degree=2, gamma=0.0001, kernel='rbf',
  max_iter=-1, probability=False, random_state=42, shrinking=True,
  tol=1e-05, verbose=False)


"""

#results for skbest with log reg
"""
             MLA Name  train f1   test f1 test f1 3*STD      time
0  LogisticRegression  0.948213  0.949068      0.105747  0:1:17.8

     steps=[('feature_selection', SelectKBest(k=23, score_func=<function f_classif at 0x000000000C487D90>)), 
     ('clf', LogisticRegression(C=10, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=1000,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=42,
          solver='saga', tol=0.001, verbose=False, warm_start=False)

"""

#results for pca with log reg
"""
             MLA Name train f1   test f1 test f1 3*STD      time
0  LogisticRegression        1  0.966603     0.0756918  0:3:43.9

     steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=0.98, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', LogisticRegression(C=100, class_weight='balanced', dual=False,
          fit_intercept=False, intercept_scaling=1, max_iter=1000,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=42,
          solver='liblinear', tol=0.001, verbose=False, warm_start=False)
"""

#results for linearsvc with pca
"""
    MLA Name train f1   test f1 test f1 3*STD      time
0  LinearSVC  0.99253  0.963105     0.0936009  0:0:19.7

     steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=0.96, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', LinearSVC(C=0.3, class_weight='balanced', dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,
     verbose=False)
"""

#results for linearsvc with skbest 
"""
MLA Name  train f1   test f1 test f1 3*STD      time
0  LinearSVC  0.992924  0.966615     0.0915804  0:0:17.7

     steps=[('feature_selection', SelectKBest(k=50, score_func=<function f_classif at 0x000000000C487D90>)), 
     ('clf', LinearSVC(C=10000, class_weight='balanced', dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=42, tol=1e-05,
     verbose=False))])]
"""

# =============================================================================
# smote at beginning of feature engineering workflow
# =============================================================================
#results for skbest with SVC and smote 
"""
  MLA Name train f1   test f1 test f1 3*STD     time
0      SVC        1  0.970114      0.160712  0:5:1.1

     steps=[('feature_selection', SelectKBest(k=40, score_func=<function f_classif at 0x000000000C1CC730>)), 
     ('clf', SVC(C=3, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=2, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=42, shrinking=True,
  tol=0.001, verbose=100)
"""

#results for skbest with gaussianNB
"""
     MLA Name  train f1   test f1 test f1 3*STD     time
0  GaussianNB  0.911498  0.913294      0.128013  0:0:7.8

     steps=[('feature_selection', SelectKBest(k=17, score_func=<function f_classif at 0x000000000C1CC730>)), 
     ('NB', GaussianNB(priors=None))])]
"""
#results for rfc
"""
                 MLA Name  train f1   test f1 test f1 3*STD       time
0  RandomForestClassifier  0.999556  0.969818      0.160853  0:36:33.4

     steps=[('clf', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=9, max_features=3, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=8,
            oob_score=True, random_state=42, verbose=0, warm_start=False)
"""

#%%
# =============================================================================
# plot learning curves for chosen model
# =============================================================================
   #%%
from matplotlib.ticker import FuncFormatter   
def plot_learning_curves_CV_reg(estimator, data = None, predictors= None, target = None,
	                         suptitle='', title='', xlabel='', ylabel='', k =10, granularity = 10, train_size= 0.9, test_size = 0.1, split = "Shuffle", random_state = 42, scale = True, center = True, doPCA=None, n_comps=None):
    # create lists to store train and validation CV scores after each full kfold step with all iterations
    train_score_CV = []
    val_score_CV = []
    #create lists to store std scores for every iteration (all folds)
    train_acc_std = []
    val_acc_std = []

    # create the split strategy
    if split == "Shuffle":
        cv = StratifiedShuffleSplit(n_splits= k , train_size= train_size, test_size= test_size, random_state = random_state)  
        max_train_samples = len(data)*train_size 
    elif split == "KFold":
        cv = StratifiedKFold(n_splits = k, shuffle=True, random_state = random_state)
        max_train_samples = len(data)-len(data)/k
    
    # create ten incremental training set sizes
    training_set_sizes = np.linspace(20, max_train_samples, granularity, dtype='int')
    # for each one of those training set sizes do the steps
    for n, i in enumerate(training_set_sizes):
        print("fitting CV folds k = {}...".format(n+1))
        # create lists to store train and validation scores for each set of kfold subloops 
        train_score = []
        val_score = []
                
        for train_i, test_i in cv.split(data[predictors], data[target]): # use kfold to "average" over the whole dataset and compute a smoothed out learning curve
            X_train, X_val = data[predictors].iloc[train_i], data[predictors].iloc[test_i]
            y_train, y_val= data[target].iloc[train_i], data[target].iloc[test_i]
            
            #scaling
            do_scale = 0
            if (scale == True) & (center == False):
                with_std = True
                with_mean = False
                do_scale = 1
            elif (scale == False) & (center == True):
                with_std = False
                with_mean = True
                do_scale = 1
            elif (scale == True) & (center == True):
                with_std = True
                with_mean = True
                do_scale = 1
                
            if do_scale == 1:
                
                #scale data now! - fit on train, transform on both
                scaler = StandardScaler(copy=True, with_mean=with_mean, with_std=with_std)
                scaled = scaler.fit(X_train)
                X_train = scaled.transform(X_train)
                X_train = pd.DataFrame(X_train)
                
                # apply scaling to test set
                X_val = scaled.transform(X_val)
                X_val = pd.DataFrame(X_val)
            
            #PCA    
            if doPCA == 1:
                pca_item =  PCA(n_components = n_comps)
                components = pca_item.fit(X_train)
                X_train = pca_item.transform(X_train)
                X_train = pd.DataFrame(X_train)
                X_val = pca_item.transform(X_val)
                X_val = pd.DataFrame(X_val)
                

            # fit the model only using that many training examples
            ravel_y_train = np.array(y_train.iloc[0:i]).ravel()
            estimator.fit(X_train.iloc[0:i, :], ravel_y_train) # ravel the vector
            
            #calculate the training accuracy only using those training examples
            train_accuracy = f1_score(y_train.iloc[0:i],
	                                    estimator.predict(X_train.iloc[0:i, :]))
	                                    
            #calculate the validation accuracy using the whole validation set
            val_accuracy = f1_score(y_val,
	                                    estimator.predict(X_val))
            
            # store the scores in their respective lists
            train_score.append(train_accuracy)
            val_score.append(val_accuracy)

        # append the stds of the cv fold values
        train_acc_std.append(np.std(train_score))
        val_acc_std.append(np.std(val_score))
        # append means of the individual scores to get final cv scores
        train_score_CV.append(np.mean(train_score))
        val_score_CV.append(np.mean(val_score))
    
    # plot learning curves on different charts
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(training_set_sizes, train_score_CV, c='gold', marker = "o")
    ax.plot(training_set_sizes, val_score_CV, c='steelblue', marker = "o")
    # plot the train / test score std ranges
    ax.fill_between(training_set_sizes, np.array(train_score_CV) - np.array(train_acc_std), np.array(train_score_CV) + np.array(train_acc_std), alpha = 0.15, color='gold')
    ax.fill_between(training_set_sizes, np.array(val_score_CV) - np.array(val_acc_std), np.array(val_score_CV) + np.array(val_acc_std), alpha = 0.15, color='steelblue')

    # format the charts to make them look nice
    fig.suptitle(suptitle, fontweight='bold', fontsize='20')
    ax.set_title(title, size=20)
    ax.set_xlabel(xlabel, size=16)
    ax.set_ylabel(ylabel, size=16)
    ax.legend(['training set', 'validation set'], fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
#    ax.set_ylim(min(train_score_CV[2:])-min(train_score_CV[2:])/2, max(val_score_CV[2:])+max(val_score_CV[2:])/10)
    ax.grid(b=True)
    for x,y in zip(training_set_sizes,val_score_CV):
        ax.annotate(str(round(y,3)), xy=(x,y), xytext=(10,-10), textcoords = 'offset points')
    
    def percentages(x, pos):
        """The two args are the value and tick position"""
        if x < 1:
            return '{:1.0f}'.format(x*100)
        return '{:1.0f}%'.format(x*100)
	
    
    def numbers(x, pos):
         """The two args are the value and tick position"""
         if x >= 1000:
             return '{:1,.0f}'.format(x)
         return '{:1.0f}'.format(x)

    x_formatter = FuncFormatter(numbers)
    ax.xaxis.set_major_formatter(x_formatter)

    y_formatter = FuncFormatter(percentages)
    ax.yaxis.set_major_formatter(y_formatter)
#%%
"""
#plot learning curves for feature selection with skbest
data = scaled_data
predictors_scaled = scaled_data.drop(target, axis=1).columns
SKB = SelectKBest(k=200)
selected_data = SKB.fit(data[predictors_scaled],data[target])
selected_data = SKB.transform(data[predictors_scaled])

labels = list(np.array(predictors_scaled)[SKB.get_support()])
selected_data = pd.DataFrame(selected_data, data.index, list(labels))
selected_data = pd.concat([selected_data, data[target]],axis=1)

MLA_final=[ # tuned algorithms with SKBest
      
  ('logreg', linear_model.LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=1000, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=42, solver='newton-cg', tol=0.0001,
          verbose=False, warm_start=False))
    ]     

for name,model in MLA_final:
    plot_learning_curves_CV_reg(model, selected_data, labels, target, suptitle = "learning curve tests", title="Tuned model CV scores for model: "+str(model.__class__.__name__), xlabel="n samples", ylabel="F1 Score", k = 4, granularity = 10, train_size = 0.9, test_size = 0.1, split = "KFold", scale = False, center = False, random_state = 42 )
"""
#%%
    #plot learning curves for feature selection with skbest
#data = scaled_data
#data = oversampling_data
data = oversampling_blsmote
predictors_scaled = data.drop(target, axis=1).columns
#SKB = SelectKBest(k=100)
#SKB = SelectKBest(k=130) # for oversampled data (136 POIs)
SKB = SelectKBest(k=43) # for SMOTE 126 POIs
selected_data = SKB.fit(data[predictors_scaled],data[target])
selected_data = SKB.transform(data[predictors_scaled])

labels = list(np.array(predictors_scaled)[SKB.get_support()])
selected_data = pd.DataFrame(selected_data, data.index, list(labels))
selected_data = pd.concat([selected_data, data[target]],axis=1)

MLA_final=[ # tuned algorithms with SKBest
      
#  ('linsvc', svm.LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
#     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
#     penalty='l2', random_state=42, tol=0.0001, verbose=False)) # lin svc for k = 100
#  ('linsvc overs', svm.LinearSVC(C=1, class_weight='balanced', dual=True, fit_intercept=True,
#     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
#     penalty='l2', random_state=42, tol=0.001, verbose=False)) # lin svc for k = 130 with oversampling (36 pois)
    ('svc smote', svm.SVC(C=3, cache_size=200, class_weight='balanced', coef0=0,
  decision_function_shape='ovr', degree=2, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=42, shrinking=True,
  tol=0.0001, verbose=False)) # svc for k = 43 with smote 126 POIs
        ]    
 

for name,model in MLA_final:
    plot_learning_curves_CV_reg(model, selected_data, labels, target, suptitle = "learning curve tests", title="Tuned model CV scores for model: "+str(model.__class__.__name__), xlabel="n samples", ylabel="F1 Score", k = 4, granularity = 10, train_size = 0.9, test_size = 0.1, split = "KFold", scale = False, center = False, random_state = 42 )


#%%
data = oversampling_blsmote
predictors = data.drop(target, axis=1).columns
    # plot learning curves for feature selection with PCA
MLA_final=[
        ('GNB', GaussianNB(),1,None)
    ]
for name,model,doPCA,n_comp in MLA_final:
    plot_learning_curves_CV_reg(model, data, predictors, target, suptitle = "learning curve tests", title="Tuned model CV scores for model: "+str(model.__class__.__name__), xlabel="n samples", ylabel="F1 Score", k = 4, granularity = 10, train_size = 0.6, test_size = 0.4, split = "KFold", scale = False, center = False, random_state = 42, doPCA=doPCA, n_comps=n_comp )

#%%
"""
#data = scaled_data
#data = oversampling_data
data = oversampling_blsmote
predictors_scaled = data.drop(target, axis=1).columns
SKB = SelectKBest(k=43) # for oversampled data SMOTE (126 pois)
#SKB = SelectKBest(k=100) k=100 with normal selected data
#SKB=SelectKBest(k=90)
#SKB = SelectKBest(k=200)
selected_data = SKB.fit(data[predictors_scaled],data[target])
selected_data = SKB.transform(data[predictors_scaled])

labels = list(np.array(predictors_scaled)[SKB.get_support()])
selected_data = pd.DataFrame(selected_data, data.index, list(labels))
selected_data = pd.concat([selected_data, data[target]],axis=1)
scores = SKB.scores_[SKB.get_support()]
#%%
skb_scores = pd.DataFrame()
skb_scores["feature names"] = labels
skb_scores["scores"] = scores
print(skb_scores.sort_values(by=["scores"],ascending=False))
"""
#%%
#"""
#data = scaled_data
data = oversampling_blsmote
predictors_scaled = data.drop(target, axis=1).columns
pca_item = PCA(n_components=None)
#pca_item = PCA(n_components=0.96)
selected_data = pca_item.fit(data[predictors_scaled],data[target])
selected_data = pca_item.transform(data[predictors_scaled])

selected_data = pd.DataFrame(selected_data, data.index)
selected_data = pd.concat([selected_data, data[target]],axis=1)
#"""
#%%
# =============================================================================
# # store the dataset for testing and export with tester.py
# =============================================================================

#clf with skbest 100 - labels in labels!
#Accuracy: 0.86860       Precision: 0.50742      Recall: 0.49600 F1: 0.50164     F2: 0.49824
#        Total predictions: 15000        True positives:  992    False positives:  963   False negatives: 1008   True negatives: 12037

#clf = svm.LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
#     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
#     penalty='l2', random_state=42, tol=0.0001, verbose=False)

# =============================================================================
# #oversampled with 34 pois manually
# #We overfit the training set!
# =============================================================================
#        Accuracy: 0.86800       Precision: 0.60638      Recall: 0.02850 F1: 0.05444     F2: 0.03521
#        Total predictions: 15000        True positives:   57    False positives:   37   False negatives: 1943   True negatives: 12963
#clf = ensemble.RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=12, max_features=1, max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=4,
#            min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=1,
#            oob_score=True, random_state=42, verbose=0, warm_start=False)

# skbest 90 from 34 oversampled pois
#        Accuracy: 0.85627       Precision: 0.46917      Recall: 0.59350 F1: 0.52406     F2: 0.56363
#        Total predictions: 15000        True positives: 1187    False positives: 1343   False negatives:  813   True negatives: 11657
#clf= LogisticRegression(C=10000, class_weight='balanced', dual=False,
#          fit_intercept=True, intercept_scaling=1, max_iter=1000,
#          multi_class='ovr', n_jobs=1, penalty='l2', random_state=42,
#          solver='newton-cg', tol=0.0001, verbose=False, warm_start=False)

# =============================================================================
# #oversampled with 136 pois manually
# =============================================================================
#        Accuracy: 0.82160       Precision: 0.40332      Recall: 0.70500 F1: 0.51310     F2: 0.61326
#        Total predictions: 15000        True positives: 1410    False positives: 2086   False negatives:  590   True negatives: 10914
#clf = svm.LinearSVC(C=1, class_weight='balanced', dual=True, fit_intercept=True,
#     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
#     penalty='l2', random_state=42, tol=0.001, verbose=False)


# =============================================================================
# #borderline smote oversampling with 252 datapoints total
# =============================================================================

# skb 80 logreg
#        Accuracy: 0.94204       Precision: 0.91285      Recall: 0.97738 F1: 0.94402     F2: 0.96376
#        Total predictions: 26000        True positives: 12706   False positives: 1213   False negatives:  294   True negatives: 11787
#clf = linear_model.LogisticRegression(C=100, class_weight='balanced', dual=False, penalty='l2', random_state=42,
#          solver='newton-cg', tol=0.0001, verbose=False, warm_start=False)

# rfc 
#        Accuracy: 0.93638       Precision: 0.93213      Recall: 0.94131 F1: 0.93670     F2: 0.93946
#        Total predictions: 26000        True positives: 12237   False positives:  891   False negatives:  763   True negatives: 12109
#clf = ensemble.RandomForestClassifier(n_jobs = 8, bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=8, max_features=2, max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=6,
#            min_weight_fraction_leaf=0.0, n_estimators=8, n_jobs=1,
#            oob_score=True, random_state=42, verbose=0, warm_start=False)

## linsvc skb 100
#        Accuracy: 0.92615       Precision: 0.88563      Recall: 0.97869 F1: 0.92984     F2: 0.95855
#        Total predictions: 26000        True positives: 12723   False positives: 1643   False negatives:  277   True negatives: 11357
#clf = svm.LinearSVC(C=1, class_weight='balanced', dual=True, fit_intercept=True,
#     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#     multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,
#     verbose=False)
  

"""best clf according to f1 score - we take that!"""
# gaussianNB with PCA None components
#        Accuracy: 0.97196       Precision: 0.99935      Recall: 0.94454 F1: 0.97117     F2: 0.95501
#        Total predictions: 26000        True positives: 12279   False positives:    8   False negatives:  721   True negatives: 12992
clf = GaussianNB()


#gaussianNB with skbest 101 components
#            Accuracy: 0.86473       Precision: 0.80420      Recall: 0.96423 F1: 0.87697     F2: 0.92732
#        Total predictions: 26000        True positives: 12535   False positives: 3052   False negatives:  465   True negatives: 9948
#clf = GaussianNB()


"""
best balanced clf so far with SMOTE at end of feature engineering workflow :) 
"""
## svc with skbest 43
#        Accuracy: 0.96335       Precision: 0.95608      Recall: 0.97131 F1: 0.96364     F2: 0.96822
#        Total predictions: 26000        True positives: 12627   False positives:  580   False negatives:  373   True negatives: 12420
#clf = svm.SVC(C=3, cache_size=200, class_weight='balanced', coef0=0,
#  decision_function_shape='ovr', degree=2, gamma=0.1, kernel='rbf',
#  max_iter=-1, probability=False, random_state=42, shrinking=True,
#  tol=0.0001, verbose=False)

## svc with pca 0.99
#        Accuracy: 0.94477       Precision: 0.91502      Recall: 0.98062 F1: 0.94668     F2: 0.96675
#        Total predictions: 26000        True positives: 12748   False positives: 1184   False negatives:  252   True negatives: 11816
#clf = svm.SVC(C=1000, cache_size=200, class_weight='balanced', coef0=0,
#  decision_function_shape='ovr', degree=2, gamma=0.0001, kernel='rbf',
#  max_iter=-1, probability=False, random_state=42, shrinking=True,
#  tol=1e-05, verbose=False)

# nusvc with skb 30
#            Accuracy: 0.93569       Precision: 0.92326      Recall: 0.95038 F1: 0.93662     F2: 0.94483
#        Total predictions: 26000        True positives: 12355   False positives: 1027   False negatives:  645   True negatives: 11973
#clf = svm.NuSVC(cache_size=200, class_weight='balanced', coef0=0.0,
#   decision_function_shape='ovr', degree=2, gamma=0.001, kernel='rbf',
#   max_iter=-1, nu=0.1, probability=False, random_state=42, shrinking=True,
#   tol=0.001, verbose=True)

## logreg with skb 23
#        Accuracy: 0.92373       Precision: 0.90728      Recall: 0.94392 F1: 0.92524     F2: 0.93636
#        Total predictions: 26000        True positives: 12271   False positives: 1254   False negatives:  729   True negatives: 11746
#clf = linear_model.LogisticRegression(C=10, class_weight='balanced', dual=False,
#          fit_intercept=True, intercept_scaling=1, max_iter=1000,
#          multi_class='ovr', n_jobs=1, penalty='l2', random_state=42,
#          solver='saga', tol=0.001, verbose=False, warm_start=False)

## logreg with pca 0.98
#        Accuracy: 0.92862       Precision: 0.88759      Recall: 0.98154 F1: 0.93220     F2: 0.96119
#        Total predictions: 26000        True positives: 12760   False positives: 1616   False negatives:  240   True negatives: 11384
#clf = linear_model.LogisticRegression(C=100, class_weight='balanced', dual=False,
#          fit_intercept=False, intercept_scaling=1, max_iter=1000,
#          multi_class='ovr', n_jobs=1, penalty='l2', random_state=42,
#          solver='liblinear', tol=0.001, verbose=False, warm_start=False)

## linearsvc with pca 0.96
#        Accuracy: 0.93677       Precision: 0.90522      Recall: 0.97569 F1: 0.93914     F2: 0.96073
#        Total predictions: 26000        True positives: 12684   False positives: 1328   False negatives:  316   True negatives: 11672
#clf = svm.LinearSVC(C=0.3, class_weight='balanced', dual=True, fit_intercept=True,
#     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#     multi_class='ovr', penalty='l2', random_state=42, tol=0.0001,
#     verbose=False)

## linearsvc with skbest 50
#        Accuracy: 0.93119       Precision: 0.89877      Recall: 0.97185 F1: 0.93388     F2: 0.95630
#        Total predictions: 26000        True positives: 12634   False positives: 1423   False negatives:  366   True negatives: 11577
#clf = svm.LinearSVC(C=10000, class_weight='balanced', dual=True, fit_intercept=True,
#     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#     multi_class='ovr', penalty='l2', random_state=42, tol=1e-05,
#     verbose=False)

# =============================================================================
# oversampling with smote at beginning of feature engineering workflow
# =============================================================================
#        Accuracy: 0.94785       Precision: 0.96060      Recall: 0.93400 F1: 0.94711     F2: 0.93920
#        Total predictions: 26000        True positives: 12142   False positives:  498   False negatives:  858   True negatives: 12502
#clf = ensemble.RandomForestClassifier(n_jobs = 8, random_state=42) # untuned 

#RFC with a little tuning
#        Accuracy: 0.95115       Precision: 0.96614      Recall: 0.93508 F1: 0.95036     F2: 0.94113
#        Total predictions: 26000        True positives: 12156   False positives:  426   False negatives:  844   True negatives: 12574
#clf = ensemble.RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=9, max_features=3, max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=8,
#            oob_score=True, random_state=42, verbose=0, warm_start=False)

# skb 40
#        Accuracy: 0.95550       Precision: 0.97285      Recall: 0.93715 F1: 0.95467     F2: 0.94408
#        Total predictions: 26000        True positives: 12183   False positives:  340   False negatives:  817   True negatives: 12660
#clf = svm.SVC(C=3, cache_size=200, class_weight='balanced', coef0=0.0,
#  decision_function_shape='ovr', degree=2, gamma=0.1, kernel='rbf',
#  max_iter=-1, probability=False, random_state=42, shrinking=True,
#  tol=0.001, verbose=100)

# =============================================================================
#    prepare feature lists for export and scoring
# =============================================================================
data = selected_data
#data = oversampling_blsmote

#data = oversampling_data
#data = scaled_data
#predictors = xgb_preds
predictors = data.drop(target,axis=1).columns
#predictors = selected_data.drop(target, axis=1).columns
features = list(predictors.values)[:]
#features = predictors[:]
features_list = features[:]
features_list.insert(0, "poi")
# store dataset in dict format
my_dataset = data.to_dict(orient="index")
#test the clf with dataset and features
test_classifier(clf, my_dataset, features_list) # we cant use xgb native api for test classifier... too bad
# and we can't use XGBClassifier from sklearn either.
#%%



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
#%%

dump_classifier_and_data(clf, my_dataset, features_list)
#%%
"""
Other Gridsearch Approach
# set up dataframe for results
MLA_columns = ["MLA Name", "train f1", "test f1", "test f1 3*STD", "train prec","test prec","train rec", "test rec", "time"]
MLA_compare = pd.DataFrame(columns=MLA_columns)
row_index = 0
#set timer for total runtime 
start_total = time.perf_counter()

# put the whole thing in cv loop over train set....
best_estimators = []
best_parameters = []

for clf, param in zip(MLA_pipe, params):
    #set timer for cv runtime
    start = time.perf_counter()
    #MLA is a list of tuples, index 0 is the name and index 1 is the algorithm
    #grid_param is a list of param_grids for the gridsearch for each estimator
    # do param search
    
    print("started with ", clf[-1][1].__class__.__name__)
    train_f1_score=[]
    train_precision_score=[]
    train_recall_score=[]
    test_auc_score=[]
    test_f1_score=[]
    test_precision_score=[]
    test_recall_score=[]

    min_target = -1
    best_params = None
    
    #get the pipeline object
    pipeline = Pipeline(clf, memory = cachedir)   
        
    for n, (train_i, test_i) in enumerate(cv_split.split(data[predictors], data[target])): # use kfold and "average" over the whole dataset, use early stopping in xgboost.train for every eval_set
        print("\nfitting CV folds k = {}...".format(n+1))    
        X_train, X_val = data[predictors].iloc[train_i], data[predictors].iloc[test_i]
        y_train, y_val= data[target].iloc[train_i], data[target].iloc[test_i]

        #create gridsearch clf
        model = model_selection.GridSearchCV(pipeline, param_grid=param, cv = cv_split, iid=False, scoring = "f1", verbose = True, return_train_score = True)
    
        # Now fit model on the data
        model.fit(X_train, y_train)
        
        # Evaluating generalization to unseen part of train set
        # use best estimator - gridsearch is autofitted
        #calculate the training accuracy 
        trainpreds = model.predict(X_train)
        train_f1 = f1_score(y_train, trainpreds)
        train_precision = precision_score(y_train,trainpreds)
        train_recall = recall_score(y_train,trainpreds)
    
        #calculate the validation accuracy 
        valpreds =   model.predict(X_val)
        test_f1 = f1_score(y_val, valpreds)
        test_precision = precision_score(y_val,valpreds)
        test_recall = recall_score(y_val,valpreds)
    
        # store the scores in their respective lists
        train_f1_score.append(train_f1)        
        train_precision_score.append(train_precision)
        train_recall_score.append(train_recall)
        
        test_f1_score.append(test_f1)        
        test_precision_score.append(test_precision)
        test_recall_score.append(test_recall)
        
        #if fold is better on test set than another we use that as our best model and parameters
        mean_target = test_f1
        if mean_target > min_target:
            min_target = mean_target
            best_algorithm = model.best_estimator_
            best_params = model.best_params_

    #store best parameters and estimators
    best_estimators.append(best_algorithm)
    best_parameters.append(best_params)
    
    #store in CV lists
    train_f1_std = (np.std(train_f1_score))
    train_f1_CV = (np.mean(train_f1_score))
    train_precision_CV = (np.mean(train_precision_score))
    train_recall_CV = (np.mean(train_recall_score))
    test_f1_std = (np.std(test_f1_score))
    test_f1_CV = (np.mean(test_f1_score))
    test_precision_CV = (np.mean(test_precision_score))
    test_recall_CV = (np.mean(test_recall_score))
    
    # store results in DF
    MLA_compare.loc[row_index, "train f1"] = train_f1_CV
    MLA_compare.loc[row_index, "test f1"] = test_f1_CV
    MLA_compare.loc[row_index, "test f1 3*STD"] = test_f1_std*3
    MLA_compare.loc[row_index, "train prec"] = train_precision_CV
    MLA_compare.loc[row_index, "test prec"] = test_precision_CV
    MLA_compare.loc[row_index, "train rec"] = train_recall_CV
    MLA_compare.loc[row_index, "test rec"] = test_recall_CV
    MLA_compare.loc[row_index, "MLA Name"] = clf[-1][1].__class__.__name__
    duration = time.perf_counter() - start
    MLA_compare.loc[row_index, "time"] = "{:.0f}:{:.0f}:{:.1f}".format(\
      duration // 3600, (duration % 3600 // 60), duration % 60)
    row_index+=1

# print and sort table:
MLA_compare.sort_values(by= ["test f1"], ascending = False, inplace=True)
rmtree(cachedir)
# print total search runtime and best params 
endtotal = time.perf_counter() - start_total

print("\nBest params for best algorithm {}:  {}, f1-score: {}".format(best_algorithm, 
      best_params, min_target))
print('Total runtime is {:.0f}:{:.0f}:{:.0f}'.format(endtotal // 3600,
      (endtotal % 3600 // 60), endtotal % 60))
print(MLA_compare)
print("\n",best_estimators)
print("\n",best_parameters)
"""
#%%
"""
# =============================================================================
# # remove variables with a high VIF
# #adding a constant is very important to calculate the correct VIF - why!?
# =============================================================================

from statsmodels.stats.outliers_influence import variance_inflation_factor 
from statsmodels.tools.tools import add_constant
# calculate_vif_ shows the features which are over the threshold and returns a new dataframe with the features removed.
def calculate_vif_(df, thresh=5):
    '''
    Calculates VIF each feature in a pandas dataframe
    A constant must be added to variance_inflation_factor or the results will be incorrect

    :param df: the pandas dataframe
    :param thresh: the max VIF value before the feature is removed from the dataframe
    :return: dataframe with features removed
    '''
    const = add_constant(df)
    cols = const.columns
    variables = np.arange(const.shape[1])
    vif_df = pd.Series([variance_inflation_factor(const.values, i) 
               for i in variables], 
              index=cols).to_frame()

    vif_df = vif_df.sort_values(by=0, ascending=False).rename(columns={0: 'VIF'})
    vif_df = vif_df.drop('const')
    vif_df = vif_df[vif_df['VIF'] > thresh]

    print('Features above VIF threshold:\n')
    print(vif_df[vif_df['VIF'] > thresh])

    col_to_drop = list(vif_df.index)

    for i in col_to_drop:
        print('Dropping: {}'.format(i))
        df.drop(columns=i, inplace = True)
    return df
#%%
scaled_data_vif = scaled_data.copy(deep=True)    

#%%
predictors = scaled_data_vif.drop(target, axis=1).columns.values
scaled_data_vif = calculate_vif_(scaled_data_vif[predictors], thresh=10) 
# only remove all above VIF threshold of 10 instead of 5
# we remove features later according to XGB feature importances and CV RMSE
#%%
scaled_data_vif.info()
#dropped columns down to 19 for thresh of 5
#dropped columns down to XXXXX for thresh of 10
"""
#%%
"""
# =============================================================================
# # CV with feature importances of XGB 
# =============================================================================
params = {
    # Parameters that we are going to tune.
    'max_depth':3,
    'min_child_weight': 1,
    'eta':.05,
    'subsample': 1,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'lambda': 1,
    'gamma' : 0,
    'nthread' : 8,
    # Other parameters
    'objective': 'binary:logistic',
    #'booster':'gblinear', # instead of gbtree for testing?
    'seed' : 42,
}
seed = 42
metrics = {'auc'} #maybe logloss?
verbose_eval = False
nfold = 10
folds = cv_split
num_boost_round = 1000
early_stopping_rounds=10

labels = target
#data = data1_cl
#data = train
data = scaled_data
predictors = data.drop(target, axis=1).columns.values
# reference the feature list for later use in the feature importance section 
features_list = predictors

# create lists to store train and validation CV scores after each full kfold step with all iterations
train_score_CV = []
val_score_CV = []
#create lists to store std scores for every iteration (all folds)
train_acc_std = []
val_acc_std = []

X_train, X_test, y_train, y_test = train_test_split(data[predictors], data[target], test_size=0.3, shuffle = True, random_state=42)

#DMatrix for every train and val set in folds
dtrain = xgboost.DMatrix(X_train, label=y_train.values, feature_names = predictors, nthread = 8)
dtest = xgboost.DMatrix(X_test, label=y_test.values, feature_names = predictors, nthread = 8)

# fit the model ####  
clf = xgboost.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtest, "Test")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval = False
            )
 
#print and store boost rounds
print("Best AUC score: {:.2f} in {} rounds".format(clf.best_score, clf.best_iteration+1))  

#calculate the training accuracy 
trainpreds = clf.predict(dtrain,  ntree_limit=clf.best_ntree_limit)
trainpreds = np.where(trainpreds > 0.5, 1, 0)  #assign binary labels

train_auc = roc_auc_score(y_train, trainpreds)
train_f1 = f1_score(y_train, trainpreds)
train_precision = precision_score(y_train,trainpreds)
train_recall = recall_score(y_train,trainpreds)

#calculate the validation accuracy 
valpreds =  clf.predict(dtest, ntree_limit=clf.best_ntree_limit)
valpreds = np.where(valpreds > 0.5, 1, 0) 

test_auc = roc_auc_score(y_test, valpreds)
test_f1 = f1_score(y_test, valpreds)
test_precision = precision_score(y_test,valpreds)
test_recall = recall_score(y_test,valpreds)

feature_importance = pd.Series(clf.get_score(importance_type='weight')).sort_values(ascending=False)
  
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())

print("roc_auc score is: {:.2f}".format(roc_auc_score(y_test, valpreds)))
print("f1_score is: {:.2f}".format(f1_score(y_test, valpreds)))
print("precision is: {:.2f}".format(precision_score(y_test,valpreds)))
print("recall is: {:.2f}".format(recall_score(y_test,valpreds)))
print(confusion_matrix(y_test,valpreds))
#%%
#feature importances (0 importance features are not included)
print(feature_importance)
#%%
#k: A threshold below which to drop features from the final data set. 
# the percentage of the most important feature's importance value
# Can cycle through threshold with CV

CVCompare_columns = ["threshold k", "train auc", "test auc", "CV train auc", "CV test auc", "CV test auc 3*STD", "CV boost_rounds", "time"]
CVCompare = pd.DataFrame(columns=CVCompare_columns)
row_index = 0

for k in  [30]:#[0,2,5,10,15,20,25,30,50,70]: 
    start = time.perf_counter() 
    fi_threshold = k  # use k for that and iterate
    
    # Get the indices of all features over the importance threshold
    important_idx = np.where(feature_importance > fi_threshold)[0]
    # Create a list of all the feature names above the importance threshold
    important_features = np.array([features_list[x] for x in important_idx])
    print("\n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance):\n", 
            important_features)
    
    # Get the sorted indexes of important features
    sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
    print("\nFeatures sorted by importance (DESC):\n", important_features[sorted_idx])
    
    #define new train set for CV loop with reduced columns according to feature importance threshold
    dtrain = xgboost.DMatrix(data[important_features[sorted_idx]], label = data[target].values, feature_names = important_features[sorted_idx], nthread = 8)
    
    print("CV with selected threshold of {}  now".format(k))
    # Run CV
    cv_results = xgboost.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        seed=seed,
        nfold=nfold,
        folds = folds,
        metrics=metrics,
        verbose_eval= verbose_eval,
        early_stopping_rounds=early_stopping_rounds
                )
    
    # get best AUC
    results_train = cv_results['train-auc-mean'].max()
    results_test = cv_results['test-auc-mean'].max()
    results_test_std = cv_results['test-auc-std'][cv_results['test-auc-mean'].idxmax()]
    boost_rounds = cv_results['test-auc-mean'].idxmax() + 1 
    
    # store results in DF
    CVCompare.loc[row_index, "threshold k"] = k
    CVCompare.loc[row_index, "train auc"] = train_auc
    CVCompare.loc[row_index, "test auc"] = test_auc
    CVCompare.loc[row_index, "CV train auc"] = results_train
    CVCompare.loc[row_index, "CV test auc"] = results_test
    CVCompare.loc[row_index, "CV test auc 3*STD"] = results_test_std*3
    CVCompare.loc[row_index, "CV boost_rounds"] = boost_rounds
    duration = time.perf_counter() - start
    print("\tRuntime of all CV folds for threshold {} was {:.0f}:{:.0f}:{:.1f}".format(k,
      duration // 3600, (duration % 3600 // 60), duration % 60))
    CVCompare.loc[row_index, "time"] = "{:.0f}:{:.0f}:{:.1f}".format(\
      duration // 3600, (duration % 3600 // 60), duration % 60)
    row_index+=1

# print and sort table:
CVCompare.sort_values(by= ["CV test auc"], ascending = False, inplace=True)
print(CVCompare)
"""
#%%
#best results with k = 30
"""
  threshold k train auc  test auc CV train auc CV test auc CV test auc 3*STD  \
0          30  0.965909  0.921053     0.993566    0.989941         0.0337279   

  CV boost_rounds     time  
0              28  0:0:0.3  
"""
#%%
xgb_preds = ['bonus', 'deferral_payments' ,'deferred_income' ,'director_fees',
 'exercised_stock_options' ,'expenses' ,'from_messages',
 'from_poi_to_this_person' ,'from_this_person_to_poi' ,'loan_advances',
 'long_term_incentive' ,'other']