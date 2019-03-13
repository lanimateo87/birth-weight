# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:31:26 2019

@author: lucas
"""

###########################################################
# Team 5 - OLS MODELS
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
# Univariate Regression Analysis
###############################################################################

########################
# Full Model
########################


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
                              out_mage +
                              out_monpre +
                              out_npvis +
                              out_fage +
                              out_feduc +
                              out_drink
                                           """,
                         data = df)


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
    The model accounts for 75.1% of the variance but some of the variables have unacceptable p-values, as well
    as very low F-statistic
    Let's consider removing these variables.

"""


########################
# Tuning the model
########################

for col in df:
    print(col)

lm_sig = smf.ols(formula = """bwght ~  
                              mage +
                              cigs +
                              drink +
                              mage + 
                              mblck +
                              fblck 
                              """,
                                  data = df)


# Fitting Results
results = lm_sig.fit()


# Printing Summary Statistics
print(results.summary())

print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")
    
    
predict = results.predict()
y_hat   = pd.DataFrame(predict).round(2)
resids  = results.resid.round(2)

'''
The model is not giving good enough predictions, let's try KNN Regressor
'''

################
# Playing with the mother's age
###############

for col in df:
    print(col)


sns.lmplot(x = 'mage',
           y = 'bwght',
           data = df,
           fit_reg = False,
           hue= 'drink', 
           palette = 'plasma')
#plt.plot([51, 51], [0, 5000], linewidth=2)
plt.plot([23,71], [2500, 2500], linewidth=2)

plt.show()


df_mother = pd.DataFrame.copy(df)


fage_high = 50
mage_high = 54
#mage_lo = 54
drink_high = 9
meduc_lo = 12
monpre_lo = 3


df_mother['out_mage_high'] = 0
df_mother['out_mage_lo'] = 0
df_mother['out_drink'] = 0
df_mother['out_meduc'] = 0
df_mother['out_fage'] = 0 
df_mother['out_monpre'] = 0


for val in enumerate(df_mother.loc[ : , 'monpre']):
    
    if val[1] <= monpre_lo:
        df_mother.loc[val[0], 'out_monpre'] = 1



for val in enumerate(df_mother.loc[ : , 'meduc']):
    
    if val[1] <= meduc_lo:
        df_mother.loc[val[0], 'out_meduc'] = 1
        


for val in enumerate(df_mother.loc[ : , 'fage']):
    
    if val[1] >= fage_high:
        df_mother.loc[val[0], 'out_fage'] = 1



for val in enumerate(df_mother.loc[ : , 'mage']):
    
    if val[1] >= mage_high:
        df_mother.loc[val[0], 'out_mage_high'] = 1
        
        
        
#for val in enumerate(df_mother.loc[ : , 'mage']):
    
 #   if val[1] < mage_lo:
#        df_mother.loc[val[0], 'out_mage_lo'] = 1
        
        
for val in enumerate(df_mother.loc[ : , 'drink']):
    
    if val[1] > drink_high:
        df_mother.loc[val[0], 'out_drink'] = 1
        

        

lm_sig = smf.ols(formula = """bwght ~  
                              out_mage_high +
                              cigs +
                              drink +
                              
                            meduc
                              """,
                                  data = df_mother)

# Fitting Results
results = lm_sig.fit()


# Printing Summary Statistics
print(results.summary())

predicts = results.predict()
y_hat = pd.DataFrame(predicts).round(2)



     

from sklearn.linear_model import LinearRegression


df_data  = df_mother.loc[ : , ['out_mage_high',
                               'drink',
                                'cigs',
                                'meduc'
                                
                                            ]]


df_target = df_mother.loc[:, 'bwght']


X_train, X_test, y_train, y_test = train_test_split(
                                               df_data,
                                               df_target,
                                               test_size = 0.1,
                                               random_state = 508)

# Prepping the Model
lr = LinearRegression()


# Fitting the model
lr_fit = lr.fit(X_train, y_train)


# Predictions
lr_pred = lr_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))





