#!/usr/bin/env python
# coding: utf-8

# # Dogecoin Price Prediction

# ### Analysis + Prediction ~ DogeCoin

# ### Author  : Saurabh

# In[1]:


pwd


# In[2]:


path='E:\\DataScience\\MachineLearning\\Dogecoin Historical Data'


# In[4]:


import os
os.listdir()


# ## Importing Libraries and Dataset

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df = pd.read_csv(path+'\DOGE-USD.csv')


# In[8]:


df.head()


# In[9]:


df1 = df.tail(135)
df1.head()


# In[10]:


df.isnull().sum()


# In[15]:


#list of features which has missing values
features_with_na=[features for features in df.columns if df[features].isnull().sum()>0]
 
#print the feature name and the percentage of missing values
for feature in features_with_na:
    print(feature, np.round(df[feature].isnull().mean(), 4),  ' % missing values')


# ## Data Visualization

# In[16]:


# A. Date v/s Volumn Graph
plt.figure(figsize=(27,7))
df.groupby('Date')['Volume'].median().plot(linewidth = 3.5, color = 'k')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title("Date vs Volume")


# In[18]:


#B. Performance of Dogecoin in the year 2022

plt.figure(figsize=(20,7))
df1.groupby('Date')['Volume'].mean().plot(linewidth = 1.5, marker ='o')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title("Date vs Volume of 2022")


# In[19]:


#C. Opening price for Dogecoin everyday throughout the year 2022

plt.figure(figsize=(20,7))
df1.groupby('Date')['Open'].mean().plot(linewidth = 2.5, color = 'm')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title("Date vs Open of 2022")


# In[20]:


#D. Maximum price achieved by Dogecoin everyday throughout the year 2022

plt.figure(figsize=(20,7))
df1.groupby('Date')['High'].mean().plot(linewidth = 2.5, color = 'c')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title("Date vs High of 2022")


# In[21]:


#E. Lowest price achieved by Dogecoin everyday throughout the year 2022

plt.figure(figsize=(20,7))
df1.groupby('Date')['Low'].mean().plot(linewidth = 2.5, color = 'b')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title("Date vs Low of 2022")


# In[22]:


#F. Closing price achieved by Dogecoin everyday throughout the year 2022

plt.figure(figsize=(20,7))
df1.groupby('Date')['Close'].mean().plot(linewidth = 2.5, color = 'g')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title("Date vs Close of 2022")


# In[23]:


#F. Closing price achieved by Dogecoin everyday throughout the year 2022

plt.figure(figsize=(20,7))
df1.groupby('Date')['Close'].mean().plot(linewidth = 2.5, color = 'g')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title("Date vs Close of 2022")


# In[24]:


#G. Lets Check the correlation between features of the dataset. How much they close to eachother.
new_df =pd.read_csv('DOGE-USD.csv', usecols = ['Open','High','Low','Close','Volume']).fillna(method='ffill')


# In[25]:


plt.figure(figsize = (8,6))
sns.heatmap(new_df.corr() , cmap = 'hot', annot = True)


# ## Prediction Models

# In[26]:


#Training and Testing Dataset Spliting using the train_test_split
X = new_df.drop('High', axis=1)
y = new_df['High']

print('Shape of X = ', X.shape)
print('Shape of y = ', y.shape)


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('Shape of X_train = ', X_train.shape)
print('Shape of y_train = ', y_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of y_test = ', y_test.shape)


# ### Random Forest Regression

# In[28]:


from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=500, criterion='mse',random_state=0)
regressor.fit(X_train, y_train)


# In[30]:


y_pred = regressor.predict(X_test)


# In[31]:


print(f'Mean Absolute Error(MAE): {metrics.mean_absolute_error(y_test, y_pred)}')
print(f'Residual Sum of Squares(MSE): {metrics.mean_squared_error(y_test, y_pred)}')
print(f'R2-Score: {metrics.r2_score(y_test, y_pred)}')


# In[32]:


ds = pd.DataFrame()
ds['High']=y_test


# In[33]:


ds['Prediction']=y_pred
ds.head(11)


# In[34]:


sns.heatmap(ds.corr(), annot = True)


# ###  Linear Regression

# In[35]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics
regr = LinearRegression()
regr.fit(X_train, y_train)


# In[36]:


y_pred = regr.predict(X_test)
print(f'Mean Absolute Error(MAE): {metrics.mean_absolute_error(y_test, y_pred)}')
print(f'Residual Sum of Squares(MSE): {metrics.mean_squared_error(y_test, y_pred)}')
print(f'R2-Score: {metrics.r2_score(y_test, y_pred)}')


# In[37]:


y_pred = regressor.predict(X_test)
ds = pd.DataFrame()
ds['High']=y_test
ds['Prediction']=y_pred
ds.head(11)


# In[38]:


sns.heatmap(ds.corr(), annot = True)


# ### Decision Tree Regressor

# In[40]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)


# In[41]:


y_pred = dtr.predict(X_test)
print(f'Mean Absolute Error(MAE): {metrics.mean_absolute_error(y_test, y_pred)}')
print(f'Residual Sum of Squares(MSE): {metrics.mean_squared_error(y_test, y_pred)}')
print(f'R2-Score: {metrics.r2_score(y_test, y_pred)}')


# In[42]:


y_pred = regressor.predict(X_test)
ds = pd.DataFrame()
ds['High']=y_test
ds['Prediction']=y_pred
ds.head(11)


# In[43]:


sns.heatmap(ds.corr(), annot = True)


# ## CONCLUSION :
#  1. Linear Regression ~ 99.86
#  2. Decision Tree Regressor ~ 99.07
#  3. Random Forest ~ 99.69
#     

# In[ ]:




