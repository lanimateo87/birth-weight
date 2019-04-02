# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:52:40 2019

@author: lucas
"""
###############################################################################
# Team EDA
###############################################################################

#importing the necessary packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#displaying all the columns when called
pd.set_option('display.max_columns', 500)

df = pd.read_excel('birthweight_feature_set.xlsx')

df.columns

df.shape

df.describe().round(2)

df.info()


###############################################################################
# Flagging missing values
###############################################################################


df.isnull().sum()

for col in df:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """

    if df[col].isnull().any():
        df['m_'+col] = df[col].isnull().astype(int)

df_dropped = df.dropna()


################################################################
# Analysing missing values from number of prenatal visits
################################################################


fig, ax = plt.subplots(figsize=(20,10))
sns.distplot(df_dropped['npvis'])

df['npvis'].describe()


'''Only 3 missing values. Taking in consideration that the factors that will
probably influence the most on the birth weight are cigaretts and drinks, and
taking in consideration that these have low correlation with number of visits
(plus the low correlation with the birthweight), we'll be inputing the
median to speed-up the analysis. Inputing the median will most likely
have little to no influence on the final model. '''

fill = df['npvis'].median()

df['npvis'] = df['npvis'].fillna(fill)


################################################################
# Analysing missing values from father's education
################################################################


fig, ax = plt.subplots(figsize=(20,10))
sns.distplot(df_dropped['feduc'])

df['feduc'].describe()

'''The larger amount of variables is concentrated around the mean and the max
value (17). To continue with the trend, we'll input the mean for the
missing values.'''

fill = df['feduc'].mean()

df['feduc'] = df['feduc'].fillna(fill)


################################################################
# Analysing missing values from mother's education
################################################################


fig, ax = plt.subplots(figsize=(20,10))
sns.distplot(df_dropped['feduc'])

'Filling with median'

fill = df['meduc'].median()

df['meduc'] = df['meduc'].fillna(fill)


###############################################################################
# General Outlier Analysis
###############################################################################


for col in df:
    print(col)


"""
Assumed Continuous/Interval Variables:
mage
monpre
npvis
fage
omaps
fmaps
cigs
drink
bwght

Assumed Ordinal:
meduc
feduc

Binary Classifiers:
male
mwhte
mblck
moth
fwhte
fblck
foth
"""


'Creating histograms to analyse the distribution'

for col in df.iloc[:, :17]:
    sns.distplot(df[col], bins = 'fd')
    plt.tight_layout()
    plt.show()


'Boxplots to check the distribution according to quantiles'

df.boxplot(column = ['male',
                     'mwhte',
                     'mblck',
                     'moth',
                     'fwhte',
                     'fblck',
                     'foth'],
                      vert = False,
                      manage_xticks = True,
                      patch_artist = False,
                      meanline = True,
                      showmeans = True
                      )
plt.show()


##############################################

df.boxplot(column = ['mage','fage'],
                     vert = False,
                     manage_xticks = True,
                     patch_artist = False,
                     meanline = True,
                     showmeans = True
                    )
plt.show()


##############################################

df.boxplot(column = ['meduc','feduc'],
                     vert = False,
                     manage_xticks = True,
                     patch_artist = False,
                     meanline = True,
                     showmeans = True
                     )
plt.show()


##############################################

df.boxplot(column = ['drink'],
                     vert = False,
                     manage_xticks = True,
                     patch_artist = False,
                     meanline = True,
                     showmeans = True
                     )
plt.show()


##############################################

df.boxplot(column = ['cigs'],
                     vert = False,
                     manage_xticks = True,
                     patch_artist = False,
                     meanline = True,
                     showmeans = True
                     )
plt.show()


##############################################

df.boxplot(column = ['omaps'],
                     vert = False,
                     manage_xticks = True,
                     patch_artist = False,
                     meanline = True,
                     showmeans = True
                     )
plt.show()


##############################################

df.boxplot(column = ['fmaps'],
                     vert = False,
                     manage_xticks = True,
                     patch_artist = False,
                     meanline = True,
                     showmeans = True
                     )
plt.show()


##############################################

df.boxplot(column = ['npvis'],
                     vert = False,
                     manage_xticks = True,
                     patch_artist = False,
                     meanline = True,
                     showmeans = True
                     )
plt.show()

##############################################

df.boxplot(column = ['monpre'],
                     vert = False,
                     manage_xticks = True,
                     patch_artist = False,
                     meanline = True,
                     showmeans = True
                     )
plt.show()


##############################################

df.boxplot(column = ['bwght'],
                     vert = False,
                     manage_xticks = True,
                     patch_artist = False,
                     meanline = True,
                    showmeans = True
                    )
plt.show()


######################################################################
# Combining outliers with thresholds gathered from qualitative analysis
######################################################################


df_2_quantiles = df.loc[:, :].quantile([0.10,
                                        0.40,
                                        0.60,
                                        0.80,
                                        0.95
                                        ])

print(df_2_quantiles)


'''Final Threshold/outlier:

Prenatal Visit low: 9
Prenatal Visit high: 15
According to qualitative analysis/research, this threshold is the healthy
interval in which a mother should go to prenatal care. Normally the minimum
threshold would be 10 visits, but let's assume 9 visits due to 1 month delay
in discovering the pregnancy. Also, more than 15 is signicant of probable
pregnancy complications.

Father's Education low: 11 (10%)
There wasn't research evidence that this could influence greatly on the
variable so this threshold was decided by outlier analysis.

Father's Age: > 45
According to the research, higher paternal age is associated with higher
incidence of premature birth, low birth weight, and others. According to the
research, 45 years should be the threshold.


Mother's Age > 35
According to research, most women have a higher change of a healthy pregnancy
between 18 and 35 years old. Putting just the higher end since the dataset
doesn't have anyone with 18 years old.

Birth Weight <= 2500
Most researchs point out that baby are considered low weight when born under
2.5kg

The other outliers were considered purely by the quantitative analysis of the
dataset.
'''


mage_high = 35
monpre_high = 4
npvis_lo = 9
npvis_hi = 15
fage_hi = 45
feduc_lo = 6
drink_high = 12
bweight_lo = 2500


########################
# Birth weight
########################

df['out_bwght'] = 0

for val in enumerate(df.loc[ : , 'bwght']):

    if val[1] <= bweight_lo:
        df.loc[val[0], 'out_bwght'] = -1


########################
# Prenatal visits
########################

df['out_npvis'] = 0

for val in enumerate(df.loc[ : , 'npvis']):

    if val[1] > npvis_hi:
        df.loc[val[0], 'out_npvis'] = 1


for val in enumerate(df.loc[ : , 'npvis']):

    if val[1] < npvis_lo:
        df.loc[val[0], 'out_npvis'] = -1


########################
# Father's age
########################

df['out_fage'] = 0

for val in enumerate(df.loc[ : , 'fage']):

    if val[1] >= fage_hi:
        df.loc[val[0], 'out_fage'] = 1


########################
# Mother's age
########################

df['out_mage'] = 0

for val in enumerate(df.loc[ : , 'mage']):

    if val[1] >= mage_high:
        df.loc[val[0], 'out_mage'] = 1


########################
# Month prenatal care started
########################

df['out_monpre'] = 0

for val in enumerate(df.loc[ : , 'monpre']):

    if val[1] >= monpre_high:
        df.loc[val[0], 'out_monpre'] = 1


########################
#Father's education in years
########################

df['out_feduc'] = 0

for val in enumerate(df.loc[ : , 'feduc']):

    if val[1] <= feduc_lo:
        df.loc[val[0], 'out_feduc'] = 1


########################
# Average drinks per week
########################

df['out_drink'] = 0

for val in enumerate(df.loc[ : , 'drink']):

    if val[1] >= drink_high:
        df.loc[val[0], 'out_drink'] = 1


###############################################################################
# Correlation Analysis
###############################################################################

df.head()
df_corr = df.loc[:, :'bwght'].corr().round(2)

print(df_corr)

df_corr.loc['bwght'].sort_values(ascending = False)

#Correlation matrix and heatmap
correlation2 = df.loc[:, :'bwght'].corr()
sns.heatmap(correlation2,
            xticklabels=correlation2.columns.values,
            yticklabels=correlation2.columns.values)
plt.show()


###############################################################################
# Scatter plots to identify patterns
###############################################################################


sns.lmplot(x = 'drink',
           y = 'bwght',
           data = df,
           fit_reg = False,
           hue= 'drink',
           palette = 'plasma'
           )
plt.show()

'''
Here it is clear that there is a association between the avg drinks per week
and the birth weight. This confirms what we saw on the correlation matrix
already.
'''

##############################################

sns.lmplot(x = 'cigs',
           y = 'bwght',
           data = df,
           fit_reg = False,
           hue= 'cigs',
           palette = 'plasma'
           )
plt.show()

'''
Same as in drinks per weeek, the cigarettes also seem to influence in the birth
weight.
'''


##############################################

sns.lmplot(x = 'cigs',
           y = 'bwght',
           data = df,
           fit_reg = False,
           hue= 'drink',
           palette = 'plasma'
           )
plt.show()

'''
Here we try to understand the relationship between the cigarretes per week,
drinks per week and birth weight. It seems that there is weak/moderate
positive correlation between drinks and cigarettes, and both having a negative
correlation with birth weight.
'''


##############################################

sns.lmplot(x = 'npvis',
           y = 'bwght',
           data = df,
           fit_reg = False,
           hue= 'drink',
           palette = 'plasma'
           )
plt.show()

''' No apparent correlation between the number of prenatal visits and
birthweight
'''


##############################################

sns.lmplot(x = 'mage',
           y = 'bwght',
           data = df,
           fit_reg = False,
           hue= 'drink',
           palette = 'plasma'
           )
# These lines will help us visually see patterns
plt.plot([35, 35], [0, 5000], linewidth=2)
plt.plot([20, 70], [2500, 2500], linewidth=2)
plt.plot([54, 54], [0, 5000], linewidth=2)
plt.show()

'''
It seems that here is a pattern of low birth weights with mothers over 54 years
old.
'''


##############################################

sns.lmplot(x = 'mage',
           y = 'bwght',
           data = df,
           fit_reg = False,
           hue= 'cigs',
           palette = 'plasma'
           )
plt.plot([35, 35], [0, 5000], linewidth=2)
plt.plot([20, 70], [2500, 2500], linewidth=2)
plt.plot([54, 54], [0, 5000], linewidth=2)
plt.show()


##############################################

sns.lmplot(x = 'fage',
           y = 'bwght',
           data = df,
           fit_reg = False,
           hue= 'drink',
           palette = 'plasma'
           )
plt.plot([23, 73], [2500, 2500], linewidth=2)

'''Although our research points out differently, the dataset doesn't
show any apparent correlation between father's age and birth weight.'''


##############################################

sns.lmplot(x = 'monpre',
           y = 'bwght',
           data = df,
           fit_reg = False,
           hue= 'drink',
           palette = 'plasma'
           )
plt.plot([0, 9], [2500, 2500], linewidth=2)
plt.show()


##############################################

sns.lmplot(x = 'meduc',
           y = 'bwght',
           data = df,
           fit_reg = False,
           hue = 'meduc',     
           palette = 'plasma'
           )
plt.show()


##############################################

"""
We can find a distinct negative correlation between birhtweight and the
use of alcohol or cigarettes.

We will probably need to tune our models a feel times to according to these
pattterns found on the scatter plots.
"""


###############################################################################
# Dataset is ready for the models
###############################################################################

df.to_excel('birthweight_team_5.xlsx')
