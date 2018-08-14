# Code by Lydia Shia
# Used reference from various Kaggle kernels 
import pandas as pd
import numpy as np
# import matplotlib as plt
import matplotlib.pyplot as plt;
import os
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import scipy as sc
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
cwd = os.getcwd()
cwd
os.chdir("/Users/lydiawawa/Documents/Dat6202/Final/Final-Project-Group4/Code/Datasets/")

pre1 = pd.read_csv("train_y.csv")
pred1_= pd.DataFrame(pre1)
pred1_.drop(pred1_.columns[1:10], axis =1, inplace=True)
pred1_.head()


train = pd.read_csv("train.csv")


train1 = pd.read_csv("trainBigC.csv")
trainO = pd.DataFrame(train)

# For visualization
trainBig = pd.DataFrame(train1)
train_ = pd.DataFrame(train)

# For Modeling and visuliazation
pets = pd.read_csv("trainK.csv")
pets = pd.DataFrame(pets)

# For Modeling
hot = pd.read_csv("train_onehot.csv")
trainHot = pd.DataFrame(hot)
trainHot.shape
#
# test = pd.read_csv("test.csv")
# test_ = pd.DataFrame(test)
# pd.options.mode.chained_assignment = None
# Id_ts=test_['ID']
# test.drop('ID',axis=1,inplace=True)

def clean(tr):
    le = preprocessing.LabelEncoder ( )
    # Save ID
    # Id_tr = tr['AnimalID']
    # tr.drop ('AnimalID', axis=1, inplace=True)

    tr['Aggressive'] = 0
    tr['At Vet'] = 0
    tr['Barn'] = 0
    tr['Behavior'] = 0
    tr['Court/Investigation'] = 0
    tr['Enroute'] = 0
    tr['Foster'] = 0
    tr['In Foster'] = 0
    tr['In Kennel'] = 0
    tr['In Surgery'] = 0
    tr['Medical'] = 0
    tr['Offsite'] = 0
    tr['Partner'] = 0
    tr['Rabies Risk'] = 0
    tr['SCRP'] = 0
    tr['Suffering'] = 0

    # OutcomeSubtype exploit
    tr.ix[tr.OutcomeSubtype == 'Aggressive', 'Aggressive'] = 1
    tr.ix[tr.OutcomeSubtype == 'At Vet', 'At Vet'] = 1
    tr.ix[tr.OutcomeSubtype == 'Barn', 'Barn'] = 1
    tr.ix[tr.OutcomeSubtype == 'Behavior', 'Behavior'] = 1
    tr.ix[tr.OutcomeSubtype == 'Court/Investigation', 'Court/Investigation'] = 1
    tr.ix[tr.OutcomeSubtype == 'Enroute', 'Enroute'] = 1
    tr.ix[tr.OutcomeSubtype == 'Foster', 'Foster'] = 1
    tr.ix[tr.OutcomeSubtype == 'In Foster', 'In Foster'] = 1
    tr.ix[tr.OutcomeSubtype == 'In Kennel', 'In Kennel'] = 1
    tr.ix[tr.OutcomeSubtype == 'In Surgery', 'In Surgery'] = 1
    tr.ix[tr.OutcomeSubtype == 'Medical', 'Medical'] = 1
    tr.ix[tr.OutcomeSubtype == 'Offsite', 'Offsite'] = 1
    tr.ix[tr.OutcomeSubtype == 'Partner', 'Partner'] = 1
    tr.ix[tr.OutcomeSubtype == 'Rabies Risk', 'Rabies Risk'] = 1
    tr.ix[tr.OutcomeSubtype == 'SCRP', 'SCRP'] = 1
    tr.ix[tr.OutcomeSubtype == 'Suffering', 'Suffering'] = 1

    global right1
    global right2
    global right3
    global right4
    global right5
    global right6
    global right7
    global right8
    global right9
    global right10
    global right11
    global right12
    global right13
    global right14
    global right15
    global right16

    right1 = pd.DataFrame (tr.groupby (['Breed'])['Aggressive'].sum ( ))
    right2 = pd.DataFrame (tr.groupby (['Breed'])['At Vet'].sum ( ))
    right3 = pd.DataFrame (tr.groupby (['Breed'])['Barn'].sum ( ))
    right4 = pd.DataFrame (tr.groupby (['Breed'])['Behavior'].sum ( ))
    right5 = pd.DataFrame (tr.groupby (['Breed'])['Court/Investigation'].sum ( ))
    right6 = pd.DataFrame (tr.groupby (['Breed'])['Enroute'].sum ( ))
    right7 = pd.DataFrame (tr.groupby (['Breed'])['Foster'].sum ( ))
    right8 = pd.DataFrame (tr.groupby (['Breed'])['In Foster'].sum ( ))
    right9 = pd.DataFrame (tr.groupby (['Breed'])['In Kennel'].sum ( ))
    right10 = pd.DataFrame (tr.groupby (['Breed'])['In Surgery'].sum ( ))
    right11 = pd.DataFrame (tr.groupby (['Breed'])['Medical'].sum ( ))
    right12 = pd.DataFrame (tr.groupby (['Breed'])['Offsite'].sum ( ))
    right13 = pd.DataFrame (tr.groupby (['Breed'])['Partner'].sum ( ))
    right14 = pd.DataFrame (tr.groupby (['Breed'])['Rabies Risk'].sum ( ))
    right15 = pd.DataFrame (tr.groupby (['Breed'])['SCRP'].sum ( ))
    right16 = pd.DataFrame (tr.groupby (['Breed'])['Suffering'].sum ( ))


    # OutcomeType
    tr.ix[tr.OutcomeType == 'Return_to_owner', 'Target'] = 1
    tr.ix[tr.OutcomeType == 'Euthanasia', 'Target'] = 0
    tr.ix[tr.OutcomeType == 'Adoption', 'Target'] = 1
    tr.ix[tr.OutcomeType == 'Transfer', 'Target'] = 1
    tr.ix[tr.OutcomeType == 'Died', 'Target'] = 0

    tr['Breed1'] = ''
    tr['Breed2'] = ''
    tr['Breed11'] = ''



    for i in range (0, tr.shape[0]):
        try:
            tr['Breed1'][i] = tr['Breed'][i].split ('/')[0]
        except IndexError:
            tr['Breed1'][i] = tr['Breed'][i]
        try:
            tr['Breed2'][i] = tr['Breed'][i].split ('/')[1]
        except IndexError:
            tr['Breed2'][i] = 'One_Breed'



    for i in range (0, tr.shape[0]):
        try:
            tr['Breed11'][i] = tr['Breed1'][i].split (' Mix')[0]
        except IndexError:
            tr['Breed11'][i] = tr['Breed1'][i]

    tr['Breed11'] = tr['Breed11'].str.upper ( )


    # Mix or not Mix
    tr['Top_Mix'] = 0

    for i in range (0, tr.shape[0]):
        if tr['Breed'][i].find (' Mix') >= 0:
            tr['Top_Mix'][i] = 1


    global a1
    global a2
    global a3
    global a4
    global a5
    global a6
    global a7
    global a8
    global a9
    global a10
    global a11
    global a12
    global a13
    global a14
    global a15
    global a16

    a1 = pd.DataFrame (tr.groupby (['Breed11'])['Aggressive'].sum ( ))
    a2 = pd.DataFrame (tr.groupby (['Breed11'])['At Vet'].sum ( ))
    a3 = pd.DataFrame (tr.groupby (['Breed11'])['Barn'].sum ( ))
    a4 = pd.DataFrame (tr.groupby (['Breed11'])['Behavior'].sum ( ))
    a5 = pd.DataFrame (tr.groupby (['Breed11'])['Court/Investigation'].sum ( ))
    a6 = pd.DataFrame (tr.groupby (['Breed11'])['Enroute'].sum ( ))
    a7 = pd.DataFrame (tr.groupby (['Breed11'])['Foster'].sum ( ))
    a8 = pd.DataFrame (tr.groupby (['Breed11'])['In Foster'].sum ( ))
    a9 = pd.DataFrame (tr.groupby (['Breed11'])['In Kennel'].sum ( ))
    a10 = pd.DataFrame (tr.groupby (['Breed11'])['In Surgery'].sum ( ))
    a11 = pd.DataFrame (tr.groupby (['Breed11'])['Medical'].sum ( ))
    a12 = pd.DataFrame (tr.groupby (['Breed11'])['Offsite'].sum ( ))
    a13 = pd.DataFrame (tr.groupby (['Breed11'])['Partner'].sum ( ))
    a14 = pd.DataFrame (tr.groupby (['Breed11'])['Rabies Risk'].sum ( ))
    a15 = pd.DataFrame (tr.groupby (['Breed11'])['SCRP'].sum ( ))
    a16 = pd.DataFrame (tr.groupby (['Breed11'])['Suffering'].sum ( ))

    return tr.shape


clean (train_)

Target = train_['Target']
train_.drop ('OutcomeSubtype', axis=1, inplace=True)
train_.drop ('Aggressive', axis=1, inplace=True)
train_.drop ('At Vet', axis=1, inplace=True)
train_.drop ('Barn', axis=1, inplace=True)
train_.drop ('Behavior', axis=1, inplace=True)
train_.drop ('Court/Investigation', axis=1, inplace=True)
train_.drop ('Enroute', axis=1, inplace=True)
train_.drop ('Foster', axis=1, inplace=True)
train_.drop ('In Foster', axis=1, inplace=True)
train_.drop ('In Kennel', axis=1, inplace=True)
train_.drop ('In Surgery', axis=1, inplace=True)
train_.drop ('Medical', axis=1, inplace=True)
train_.drop ('Offsite', axis=1, inplace=True)
train_.drop ('Partner', axis=1, inplace=True)
train_.drop ('Rabies Risk', axis=1, inplace=True)
train_.drop ('SCRP', axis=1, inplace=True)
train_.drop ('Suffering', axis=1, inplace=True)
train_.drop ('OutcomeType', axis=1, inplace=True)
train_.drop ('Target', axis=1, inplace=True)


train_.drop ('Breed1', axis=1, inplace=True) #First breed
train_.drop ('Breed2', axis=1, inplace=True) #Second breed if any
train_.rename(columns = {'Breed11':'Main Breed'}, inplace=True)



# trainBig.drop ('Breed1', axis=1, inplace=True) #First breed
# trainBig.drop ('Breed2', axis=1, inplace=True) #Second breed if any
trainBig.rename(columns = {'Breed11':'Main Breed'}, inplace=True)


trainBig['ageperiod'].replace ('year', 'years', inplace=True)
trainBig['ageperiod'].replace ('day', 'days', inplace=True)
trainBig['ageperiod'].replace ('month', 'months', inplace=True)
trainBig['ageperiod'].replace ('week', 'weeks', inplace=True)
trainBig['ageperiod'].replace ('year', 'years', inplace=True)
trainBig['ageperiod'].replace ('day', 'days', inplace=True)
trainBig['ageperiod'].replace ('month', 'months', inplace=True)
trainBig['ageperiod'].replace ('week', 'weeks', inplace=True)


train_.head()
# Drop examine subtype by breed -------------
#
# a1 = a1.sort_values(by=['Aggressive'], ascending=False)
# a2.sort_values(by=['At Vet'], ascending=False)




# Drop columns before merging ---------------

# col = [7,8]
# col = list(df.columns)[0:7]
# train_.columns[0:7]
train_.drop(train_.columns[1:10], axis =1, inplace=True)


# Encoding AgeUponOutcome_lad ----------------

# Manual Mapping of categorical column using dictionary

# specify the mapping
# size_mapping = {'days':1, 'weeks': 2,'months': 3, 'years': 4}

# apply the mapping
# train_['AgeUnit'] = train_['AgeUnit'].map(size_mapping)


# Ecoding Main Breed ----------------
class_le = LabelEncoder()
train_["BreedName"] = train_["Main Breed"].copy()
train_['Main Breed'] = class_le.fit_transform(train_['Main Breed'].values)

# y_inv = class_le.inverse_transform(train_['Main Breed'])
#
# sc.stats.itemfreq(y_inv)

train_.head()



# train1= train1.iloc[0:0]

# del train1


# Combine with previous preprocessing dataset ----------------

trainC = pd.merge(train_,pred1_, on='AnimalID', how='outer')

trainC.shape
trainC.columns

train_.shape; pred1_.shape





# Check missing --------------------
trainC.isnull().values.any()
null_data = trainC[trainC.isnull().any(axis=1)]
trainC.info(null_counts=True)
trainC.isnull().sum()


# Remove missing:

trainNmiss= trainC.dropna()

trainNmiss.shape
trainNmiss.isnull().values.any()


# Drop repeated variables
# trainNmiss.drop ('Name_bool', axis=1, inplace=True)
# trainNmiss.drop ('sprayed', axis=1, inplace=True)

# trainNmiss.drop ('MainBreed', axis=1, inplace=True)
trainNmiss.drop ('Top_Mix', axis=1, inplace=True)

trainNmiss.drop ('MainBreedMix', axis=1, inplace=True)
# trainNmiss.rename(columns = {'MainBreed':'MainBreedMix'}, inplace=True)




# trainC = train_+pred1_
# pred1_.shape
# Write to Excel -------------------------
trainNmiss.to_csv('trainK',encoding='utf-8', index=False)

# tips = sns.load_dataset("tips")
# tips.head()


# Data Visualization ----------------------
print(plt.style.available)
plt.style.use('seaborn-poster')

plt.figure(1)

plt.subplot (411)
a1_l = a1['Aggressive'].tolist()
a1_i = a1.index
a1_i = a1_i.tolist()
a1_i = a1_i[0:10]
a1_l = a1_l[0:10]

a1.reset_index(level=0, inplace=True)
a1g = a1.iloc[0:10,0:2]


sns.distplot(x="Breed11", y="Aggressive", data=a1g)

plt.bar(a1_i, a1_l)
plt.xticks(rotation=90)
plt.show()

# sns.distplot(a1_i, a1_l)
# sns.show()
# a1['Aggressive']



trainNmiss['outcome'].value_counts()



# y_inv.value_counts()
# train_['Main Breed'].value_counts()

# train_[:1:]

# df.sort_values(by=['col1', 'col2'])



#OutCome Subtype Analysis --------
trainO.shape
trainO['OutcomeSubtype'].isnull().sum ( )

# 13612/26729

# Correlation
pets = pd.read_csv("trainK.csv")
pets = pd.DataFrame(pets)

pets.columns
pets.drop ('Aggressive', axis=1, inplace=True)
pets.drop ('At Vet', axis=1, inplace=True)
pets.drop ('Barn', axis=1, inplace=True)
pets.drop ('Behavior', axis=1, inplace=True)
pets.drop ('Court/Investigation', axis=1, inplace=True)
pets.drop ('Enroute', axis=1, inplace=True)
pets.drop ('Foster', axis=1, inplace=True)
pets.drop ('In Foster', axis=1, inplace=True)
pets.drop ('In Kennel', axis=1, inplace=True)
pets.drop ('In Surgery', axis=1, inplace=True)
pets.drop ('Medical', axis=1, inplace=True)
pets.drop ('Offsite', axis=1, inplace=True)
pets.drop ('Partner', axis=1, inplace=True)
pets.drop ('Rabies Risk', axis=1, inplace=True)
pets.drop ('SCRP', axis=1, inplace=True)
pets.drop ('Suffering', axis=1, inplace=True)
pets.drop ('AnimalID', axis=1, inplace=True)
pets.drop ('BreedName', axis=1, inplace=True)
pets.drop ('ageperiod', axis=1, inplace=True)
pets.drop ('outcome', axis=1, inplace=True)
pets.drop ('Target', axis=1, inplace=True)

# Compute the correlation matrix
corr = pets.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask,  cmap=cmap, vmin=corr.values.min(), vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
annot=True, annot_kws = {"size": 6.5})

# plt.savefig('corr1.png')
plt.show()
plt.close()

trainBig.head()
# Explore variable impact to outcome
ct1 = pd.crosstab(trainBig.OutcomeType, trainBig.AnimalType)
ct2 = pd.crosstab(trainBig.OutcomeType, trainBig.SexuponOutcome)
ct3 = pd.crosstab(trainBig.OutcomeType, trainBig.ageperiod)
ct4 = pd.crosstab(trainBig.OutcomeType, trainBig.MixBreed)
ct5 = pd.crosstab(trainBig.OutcomeType, trainBig.HaveName)
ct6 = pd.crosstab(trainBig.OutcomeType, trainBig.fertility)
ct7 = pd.crosstab(trainBig.OutcomeType, trainBig.MixColor)
ct8 = pd.crosstab(trainBig.OutcomeType, trainBig.colorC)
# stacked = ct1.stack ( ).reset_index ( ).rename (columns={0: 'value'})
# sns.barplot(x=stacked.OutcomeType, y=stacked.value, hue=stacked.AnimalType)

print(plt.style.available)
plt.style.use('seaborn-muted')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rc('legend',**{'fontsize':18})

plt.fig.subplots_adjust()
fig = plt.figure()
fig.subplots_adjust(bottom = 0)
fig.subplots_adjust(top = 1)
fig.subplots_adjust(right = 1)
fig.subplots_adjust(left = 0)

# plt.rcParams['title_fontsize'] = 19

plt.figure(1)





ct1.plot.bar(stacked=True)
# plt.legend(title='AnimalType')
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 18)
leg = ax.legend()
leg.set_title('AnimalType',prop={'size':12})
# plt.tick_params(labelsize=18)
# ax.tick_params(axis = 'both', which = 'minor', labelsize = 16)
plt.show()


ct2.plot.bar(stacked=True)
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 18)
leg = ax.legend()
leg.set_title('SexuponOutcome',prop={'size':16})
plt.show()


#Age by unit of time (year, month, week, day)
ct3.plot.bar(stacked=True)
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 18)
leg = ax.legend()
leg.set_title('Age Unit',prop={'size':16})
plt.show()

#Age boxplot
ax = sns.boxplot(x="OutcomeType", y="age", data=trainBig).set_title('Age in Days')

#Mix Breed Yes or No
ct4.plot.bar(stacked=True)
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 18)
leg = ax.legend()
leg.set_title('Mix Breed Yes(1) or No(0)',prop={'size':16})
plt.show()

#Have a name yes or no
ct5.plot.bar(stacked=True)
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 18)
leg = ax.legend()
leg.set_title('Have Name Yes(1) or No(0)',prop={'size':16})
plt.show()

#Fertility
ct6.plot.bar(stacked=True)
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 18)
leg = ax.legend()
leg.set_title('Spayed(1) Intact(0) Unknown(2)',prop={'size':16})
plt.show()

#Mix Color
ct7.plot.bar(stacked=True)
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 18)
leg = ax.legend()
leg.set_title('Mixed Color Yes(1) or No(0)',prop={'size':16})
plt.show()

#By Color

ct9 = ct8.iloc[0:10,0:10]
ct9.plot.bar(stacked=True)
ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 14)
leg = ax.legend()
leg.set_title('Color',prop={'size':16})
plt.legend(title="Color",loc="center left",bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.show()
