#!/usr/bin/env python
# coding: utf-8

# In[29]:


# Fraud Detection using Machine Learning 
#     - Data Cleaning, shuffling, scaling and outliers handling
#     - Balancing the data for equated weightage on both genuine and fraud classes
#     - Fitting a Random Forest Classifier to train the model on genuine and fraudulent transactions

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings

warnings.filterwarnings('ignore') #ignore warnings message

df = pd.read_csv(r"file:///E:\DS - Sohini\Class 24 - Naive Bayes Model\creditcard.csv")
df.head()


# In[30]:


df.shape


# In[31]:


df.isnull().sum().max()


# In[32]:


# the classes are heavily skewed we need to solve the issue later
print('No Frauds', round(df['Class'].value_counts()[0]/len(df)*100,2), '% of the dataset')


# In[33]:


print('Frauds', round(df['Class'].value_counts()[1]/len(df)*100,2), '% of the dataset')


# In[34]:


sns.countplot(x='Class', data=df, palette='rainbow')
plt.title('Class distribution \n (0:no fraud, 1: fraud)',fontsize=14)


# In[35]:


fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of transaction amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of transaction time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()


# In[36]:


# Since most of the data is already scaled we should scale that are left to scale ( amount and time)
from sklearn.preprocessing import RobustScaler

# Robustscaler is less prone to outliers
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)


# In[37]:


scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

# amount and time are scaled
df.head()


# In[38]:


from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold


# In[39]:


x = df.drop('Class', axis=1)
y = df['Class']


# In[40]:


sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)


# In[41]:


for train_index, test_index in sss.split(x,y):
    original_xtrain, original_xtest = x.iloc[train_index], x.iloc[test_index]
    original_ytrain, original_ytest = x.iloc[train_index], x.iloc[test_index]


# In[42]:


# lets shuffle the data before creating the subsamples

df = df.sample(frac=1)

# amount of fraud classes 492 rows
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0] [0:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])


# In[43]:


# Shuffle fataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)


# In[44]:


new_df


# In[45]:


sns.countplot(x='Class', data=new_df, palette='copper_r')
plt.title('equally distributed classes',fontsize=14)
plt.show()


# In[46]:


# Correlation matrix

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))

corr  = df.corr()
sns.heatmap(corr, cmap = 'coolwarm_r', annot_kws={'size' : 20}, ax=ax1)
ax1.set_title("Imbalanced Correlation matrix \n (don't use for reference)", fontsize=14)

sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size' : 20}, ax=ax2)
ax2.set_title("subsample Correlation matrix \n (use for reference)", fontsize=14)

plt.show()


# In[47]:


f, axes = plt.subplots(ncols=4, figsize=(20,4))

# negative correlation with our class ( the lower our feature value the more likely it willl be a fraud txn)
sns.boxplot(x="Class", y="V17", data=new_df, palette="viridis", ax=axes[0])
axes[0].set_title("V17 vs Class negative Corrleation")

sns.boxplot(x="Class", y="V14", data=new_df, palette="terrain", ax=axes[1])
axes[1].set_title("V14 vs Class negative Corrleation")

sns.boxplot(x="Class", y="V12", data=new_df, palette="crest", ax=axes[2])
axes[2].set_title("V12 vs Class negative Corrleation")

sns.boxplot(x="Class", y="V10", data=new_df, palette="cool", ax=axes[3])
axes[3].set_title("V10 vs Class negative Corrleation")




# In[48]:


from scipy.stats import norm

f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,6))

v14_fraud_dist = new_df['V14'].loc[new_df['Class']==1].values
sns.distplot(v14_fraud_dist, ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('v14 distribution \n (fraud txn)', fontsize=14)


v12_fraud_dist = new_df['V12'].loc[new_df['Class']==1].values
sns.distplot(v12_fraud_dist, ax=ax2, fit=norm, color='#FB8861')
ax1.set_title('v12 distribution \n (fraud txn)', fontsize=14)

v10_fraud_dist = new_df['V10'].loc[new_df['Class']==1].values
sns.distplot(v10_fraud_dist, ax=ax3, fit=norm, color='#FB8861')
ax1.set_title('v10 distribution \n (fraud txn)', fontsize=14)

plt.show()


# In[49]:


# v14 removing outliers (highest negative correlated with labels)
v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values

q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud,75)

print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))

v14_iqr = q75 - q25
print('iqr : {}'.format(v14_iqr))


# In[50]:


#lf = q1 - 1.5*IQR
#uf = q3 + 1.5*IQR


# In[51]:


v14_cutoff = v14_iqr*1.5
v14_lower, v14_upper = q25 - v14_cutoff, q75 + v14_cutoff

print('Cutoff : {}'.format(v14_cutoff))
print('v14 lower : {}'.format(v14_lower))
print('v14 upper : {}'.format(v14_upper))


# In[52]:


outliers = [x for  x in v14_fraud if x < v14_lower or x > v14_upper]
print('Feature v14 outliers for fraud  cases : {}'.format(len(outliers)))
print('v14 outliers : {}'.format(outliers))


# In[54]:


new_df2 = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)


# In[55]:


new_df2.shape


# In[57]:


# v12 removing outliers (highest negative correlated with labels)
v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values

q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud,75)

print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))

v12_iqr = q75 - q25
print('iqr : {}'.format(v12_iqr))


# In[58]:


v12_cutoff = v12_iqr*1.5
v12_lower, v12_upper = q25 - v12_cutoff, q75 + v12_cutoff

print('Cutoff : {}'.format(v12_cutoff))
print('v12 lower : {}'.format(v12_lower))
print('v12 upper : {}'.format(v12_upper))


# In[59]:


outliers = [x for  x in v12_fraud if x < v12_lower or x > v12_upper]
print('Feature v12 outliers for fraud  cases : {}'.format(len(outliers)))
print('v12 outliers : {}'.format(outliers))


# In[60]:


new_df2 = new_df2.drop(new_df2[(new_df2['V12'] > v12_upper) | (new_df2['V12'] < v12_lower)].index)


# In[61]:


new_df2.shape


# In[66]:


# v10 removing outliers (highest negative correlated with labels)
v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values

q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud,75)

print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))

v10_iqr = q75 - q25
print('iqr : {}'.format(v10_iqr))


# In[67]:


v10_cutoff = v10_iqr*1.5
v10_lower, v10_upper = q25 - v10_cutoff, q75 + v10_cutoff

print('Cutoff : {}'.format(v10_cutoff))
print('v10 lower : {}'.format(v10_lower))
print('v10 upper : {}'.format(v10_upper))


# In[68]:


outliers = [x for  x in v10_fraud if x < v10_lower or x > v10_upper]
print('Feature v10 outliers for fraud  cases : {}'.format(len(outliers)))
print('v10 outliers : {}'.format(outliers))


# In[71]:


new_df2 = new_df2.drop(new_df2[(new_df2['V10'] > v10_upper) | (new_df2['V10'] < v10_lower)].index)


# In[72]:


new_df2.shape


# In[78]:


f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,6))

colors = ['#B3F9C5', '#f9c5b3']
# Boxplot with outliers removed
# Feature V14
sns.boxplot(x='Class' , y='V14', data=new_df2, ax=ax1, palette=colors)
ax1.set_title("V14 Feature \n Reduction of outliers", fontsize=14)
ax1.annotate('Fewer extreme \n outliers', xy = (0.98, -17.5), xytext=(0,-12), fontsize=14)

# Feature V12
sns.boxplot(x='Class' , y='V12', data=new_df2, ax=ax2, palette=colors)
ax2.set_title("V12 Feature \n Reduction of outliers", fontsize=14)
ax2.annotate("Fewer extreme \n outliers", xy = (0.98, -17.3), xytext=(0,-12), fontsize=14)

# Feature V10
sns.boxplot(x='Class' , y='V10', data=new_df2, ax=ax3, palette=colors)
ax3.set_title("V10 Feature \n Reduction of outliers", fontsize=14)
ax3.annotate("Fewer extreme \n outliers", xy = (0.98, -14.3), xytext=(0,-12), fontsize=14)


# In[79]:


from sklearn.model_selection import train_test_split

train_target = new_df2['Class']
train = new_df2.drop(['Class'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(train, train_target, random_state=0)


# In[91]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc

from scipy.stats import randint
import pickle
#import xgboost as xgb


# In[82]:


class_weight = dict({0:1, 1:100})

random_search = {'criterion' : ['entropy', 'gini'], 
                 'max_depth' : [2,3,4,5,6,7,10],
                'min_samples_leaf' : [4,6,8],
                'min_samples_split' : [5,7,10],
                'n_estimators' : [300]}

clf = RandomForestClassifier(class_weight = class_weight)
model = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 10, 
                          cv=4, verbose=1, random_state=101, n_jobs=-1)

model.fit(x_train, y_train)


# In[86]:


y_pred = model.predict(x_test)
print(classification_report(y_test,y_pred))


# In[87]:


y_score = model.predict_proba(x_test)[:,1]


# In[92]:


fpr, tpr, _ = roc_curve(y_test, y_score)

plt.title('random forest ROC curve : ')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr, tpr)
plt.plot((0,1), ls='dashed', color='black')
plt.show()
print('Area under curve AUC : ', auc(fpr, tpr))


# In[ ]:




