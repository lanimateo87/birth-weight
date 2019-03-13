# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:43:54 2019

@author: lucas
"""

############################################################
# Team 5 - KNN MODELS
############################################################

# Importing necessary packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf 
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor
import sklearn.metrics
from sklearn.model_selection import cross_val_score

# Importing new libraries
from sklearn.tree import DecisionTreeRegressor # Regression trees
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
import pydotplus # Interprets dot objects


file = 'birthweight_team_5.xlsx'
df= pd.read_excel(file)


###############################################################################
# KN-Neighbors - MODEL 1
###############################################################################

for col in df:
    print(col)

df_data  = df.loc[ : , ['mage',
                        'meduc',
                        'drink',
                        'mblck',
                        'moth',
                        'cigs',
                        ]]




df_target = df.loc[:, 'bwght']


X_train, X_test, y_train, y_test = train_test_split(
                                               df_data,
                                               df_target,
                                               test_size = 0.1,
                                               random_state = 508)



########################
# Using KNN  On Our Optimal Model 
########################

# Exact loop as before
training_accuracy = []
test_accuracy = []


neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))



plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)




########################
# Create a model object
########################

# Creating a regressor object
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 7)



# Checking the type of this new object
type(knn_reg)


# Teaching (fitting) the algorithm based on the training data
knn_reg.fit(X_train, y_train)



# Predicting on the X_data that the model has never seen before
y_pred = knn_reg.predict(X_test)



# Printing out prediction values for each test observation
print(f"""
Test set predictions:
{y_pred}
""")


# Calling the score method, which compares the predicted values to the actual
# values
y_score_train = knn_reg.score(X_train, y_train)

y_score = knn_reg.score(X_test, y_test)



# The score is directly comparable to R-Square
print(f"Test score: {y_score}")

print(f"Train Score: {y_score_train}")



################################################
# Storing Model Predictions and Summary
################################################

# We can store our predictions as a dictionary.
model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'KNN_Predicted': knn_reg_optimal_pred,
                                     'OLS_Predicted': lr_pred})



model_predictions_df.to_excel("Ames_Model_Predictions.xlsx")


###############################################################################
# MODEL 2
###############################################################################

for col in df:
    print(col)

df_data  = df.loc[ : , ['mage',
                        'meduc',
                        'drink',
                        'mblck',
                        'moth',
                        'cigs',
                        ]]




df_target = df.loc[:, 'bwght']


X_train, X_test, y_train, y_test = train_test_split(
                                               df_data,
                                               df_target,
                                               test_size = 0.1,
                                               random_state = 508)



########################
# Using KNN  On Our Optimal Model 
########################

# Exact loop as before
training_accuracy = []
test_accuracy = []


neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))



plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)




########################
# Create a model object
########################

# Creating a regressor object
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 7)



# Checking the type of this new object
type(knn_reg)


# Teaching (fitting) the algorithm based on the training data
knn_reg.fit(X_train, y_train)



# Predicting on the X_data that the model has never seen before
y_pred = knn_reg.predict(X_test)



# Printing out prediction values for each test observation
print(f"""
Test set predictions:
{y_pred}
""")


# Calling the score method, which compares the predicted values to the actual
# values
y_score_train = knn_reg.score(X_train, y_train)

y_score = knn_reg.score(X_test, y_test)



# The score is directly comparable to R-Square
print(f"Test score: {y_score}")

print(f"Train Score: {y_score_train}")




###############################################################################
# MODEL 3
###############################################################################



df_data  = df_mother.loc[ : , ['out_mage_high',
                                            'drink',
                                            'cigs',
                                            ]]




df_target = df_mother.loc[:, 'bwght']


X_train, X_test, y_train, y_test = train_test_split(
                                               df_data,
                                               df_target,
                                               test_size = 0.1,
                                               random_state = 508)



########################
# Using KNN  On Our Optimal Model 
########################

# Exact loop as before
training_accuracy = []
test_accuracy = []


neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))



plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)




########################
# Create a model object
########################

# Creating a regressor object
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 29)



# Checking the type of this new object
type(knn_reg)


# Teaching (fitting) the algorithm based on the training data
knn_reg.fit(X_train, y_train)



# Predicting on the X_data that the model has never seen before
y_pred = knn_reg.predict(X_test)



# Printing out prediction values for each test observation
#print(f"""
#Test set predictions:
#{y_pred}
#""")


# Calling the score method, which compares the predicted values to the actual
# values
y_score_train = knn_reg.score(X_train, y_train)

y_score = knn_reg.score(X_test, y_test)

print(y_score_train)
print(y_score)

















