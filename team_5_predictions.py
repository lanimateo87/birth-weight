# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:52:40 2019

@author:
TEAM 5
   Ayodeji Sobanke
   Bhagyashree Mulkalwar
   Daniela Santacruz Aguilera
   Lani Mateo
   Lucas Barros
   Ronak Patel

Objective: To identify genetic and non-genetic characteristics that are
associated with low birthweight using Linear Regression Modeling.

"""

###############################################################################
# Linear Modeling
###############################################################################


# Importing libraries
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression


#Importing out file from excel with the new variables:
df= pd.read_excel('birthweight_team_5.xlsx')

###############################
# OLS Linear regression
##############################

'Full Model:'

for col in df:
    print(col)

lm_full = smf.ols(formula = """bwght ~ mage +
                                       meduc +
                                       monpre +
                                       npvis +
                                       fage +
                                       feduc +
                                       omaps +
                                       fmaps +
                                       cigs +
                                       drink +
                                       male +
                                       mwhte +
                                       mblck +
                                       moth +
                                       fwhte +
                                       fblck +
                                       foth +
                                       m_meduc +
                                       m_npvis +
                                       m_feduc +
                                       out_bwght+
                                       out_mage +
                                       out_monpre +
                                       out_npvis +
                                       out_fage +
                                       out_feduc +
                                       out_drink+
                                       out_cigs
                                       """,
                                       data = df
                                       )


# Fitting Results
results = lm_full.fit()

# Printing Summary Statistics
print(results.summary())

print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")

"""
The model accounts for 77.3% of the variance but some of the variables have
unacceptable p-values. Let's consider removing these variables.
"""

####################
# Tuning the Model
####################

lm_full = smf.ols(formula = """bwght ~ mage+
                                       cigs+
                                       drink+
                                       mwhte +
                                       mblck +
                                       moth +
                                       fwhte +
                                       fblck +
                                       foth
                                       """,
                                       data = df
                                       )


# Fitting Results
results = lm_full.fit()

# Printing Summary Statistics
print(results.summary())

print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")

"""
This  model accounts for 70.6% of the variance, we're going to try to
improve our model adding and taking variables, based on research.
"""


#############
# Best Model
#############

"""
1) According to our research African-American and Latino mothers have
higher chance to have a low weighted baby than White mothers, but there
is no actual scientific research that supports this theory, and
scientist associate this more to socioeconomic factors, which is why
we decided to remove this variables from our model.
2) Removing race from our model does not affect our R squared
3) Adding mother's education to our model improves our R squared and
it is significant.
"""

lm_sig = smf.ols(formula = """bwght ~ mage+
                                      drink+
                                      cigs +
                                      meduc+
                                      out_mage
                                      """,
                                      data = df
                                      )


# Fitting Results
results = lm_sig.fit()

# Printing Summary Statistics
print(results.summary())

print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")

"""
This  model accounts for 72.7% of the variance, it is our best model so
far, with all the variables being significant.
"""


##########################################
# Generalization using Train/Test Split
##########################################
for col in df:
    print(col)

"""
We tested our significant model amongst other models where our R squared
was high, however our train and test prediction was very low and the
gap between them was very high.
We used the variables that we got in our best model for the Trai/Test Split
"""

#Separating the features from the independent variable
df_data= df.loc[ :,['mage',
                    'drink',
                    'cigs',
                    'meduc',
                    'out_mage'
                    ]]


df_target = df.loc[:, 'bwght']



X_train, X_test, y_train, y_test = train_test_split(
                                               df_data,
                                               df_target,
                                               test_size = 0.10,
                                               random_state = 508)


# Let's check to make sure our shapes line up.

# Training set
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)


########################
# OLS prediction
########################

# Prepping the Model
lr = LinearRegression()


# Fitting the model
lr_fit = lr.fit(X_train, y_train)


# Predictions
lr_pred = lr_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))

#Storing predictions from our final model:
model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'OLS_Predicted': lr_pred})


model_predictions_df.to_excel("Team_5_Predictions.xlsx")


########################
##Comparing to KNN
########################

# Creating two lists, one for training set accuracy and the other for test
# set accuracy
training_accuracy = []
test_accuracy = []


# Building a visualization to check to see  1 to 50
neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)

    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))

    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


# Plotting the visualization
fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()


# Optimal number of neighbors?
print(test_accuracy.index(max(test_accuracy)))


#Using the optimal number +1 due to the count of indexes: 8 neighbors
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 8)


# Checking the type of this new object
type(knn_reg)


# Fitting based on the training data
knn_reg.fit(X_train, y_train)


# Predicting on the X_data
y_pred = knn_reg.predict(X_test)


# Printing out prediction values for each test observation
print(f"""Test set predictions:
      {y_pred}""")

# Calling the score method, which compares the predicted values to the actual
# values:
y_score_test = knn_reg.score(X_test, y_test)
y_score_train = knn_reg.score(X_train, y_train)


print(y_score_test)
print(y_score_train)
"""There's a big difference between my train and my test score,
   it didn't perform well, we get a better testing score using OLS Linear
   Regression.
   We use the model created with OLS.
"""
