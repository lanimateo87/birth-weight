# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 23:45:52 2019
@author: AYODEJI
Working Directory:
C:\Users\AYODEJI\Desktop\PYCOURSE\MACHINE LEARNING\ML SCRIPTS
Purpose:
    To review new birthweight set.
"""

# Loading Libraries and importing data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf # regression modeling

file = 'birthweight_feature_set.xlsx'

birthweightfeature = pd.read_excel(file)
df = birthweightfeature

# Getting general information about the dataset 
df.columns # Checking for Column names

print(df.head()) # Displaying the first rows of the DataFrame

df.shape  # Checking dimensions of the DataFrame

df.info()  # Information about each variable

df.describe().round(2) # Descriptive statistics

df.sort_values('bwght', ascending = False) # Sorting birthweight in ascending order

# Missing Values

print(df.isnull().sum()) # Checking for missing values per column

print(df.isnull().sum().sum()) # Checking for total number of missing value

# Creating columns that are 0s if a value was not missing and 1 if a value is missing
print(df.isnull().sum())
for col in df:
        if df[col].isnull().any():
            df['m_'+col] = df[col].isnull().astype(int)

# To use median value to fill in the missing values
df_new = df  # let's first create a new dataframe

for col in df_new.columns:
    if (df_new[col].isnull().any()) == True:
        df_new[col] = df_new[col].fillna(df_new[col].median())
         
# Confirming if the missing values were correctly imputed
df_new.info()
print(df_new.describe().round(2))
print(df_new.isnull().sum().sum())

# Further analysis of the dependent variable: Birth Weight (bwght) using 1. Descriptive statistics 
df_new['bwght'].describe()  # Descriptive statistics
df_new['bwght'].sort_values(ascending = True).head(30) # Sorting birthweight in ascending order viewing the first 30 results

# Using 2. Distribution
sns.distplot(df_new['bwght'])

# CORRELATION ANALYSIS VISUAL APPROACH
df_new_corr = df_new.corr()

# To do the graph of the heatmap
fig, ax=plt.subplots(figsize=(10,10))
sns.set(font_scale=2)
sns.heatmap(df_new_corr,
            cmap = 'Blues',
            square = True,
            annot = False,
            linecolor = 'black',
            linewidths = 0.5)

#plt.savefig('correlation_matrix_all_var')
plt.show()

# CORRELATION ANALYSIS STATISTICAL APPROACH
df_new_corr['bwght'].sort_values(ascending=True).head(10) # Negative correlation

df_new_corr['bwght'].sort_values(ascending=False).head(10) # Positive correlation

# CORRELATION ANALYSIS GRAPHICAL APPROACH #Plotting graph for all variables that we think are interesting after looking the heatmap and the numerical correlations;
sns.set()
cols = ['bwght', 'drink', 'cigs', 'omaps', 'mage', 'fage', 'moth', 'feduc', 'foth', 'fwhte']
sns.pairplot(df_new[cols], height= 2.5)
plt.show();

# Numerical Variables Distribution; plotting histogram for all the variables to have a graphical idea of their distribution.
df_new.hist(figsize=(16, 20), bins=50, xlabelsize=12, ylabelsize=12)







