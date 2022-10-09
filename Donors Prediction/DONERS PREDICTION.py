#!/usr/bin/env python
# coding: utf-8

# ## DONERS PREDICTION

# In[1]:


## Author : Saurabh Kumar
## date : 1st-oct


# In[2]:


pwd


# In[3]:


path ='E:\\DataScience\\MachineLearning\\Donors Prediction'


# In[4]:


import os
os.listdir()


# In[5]:


import pandas as pd
df = pd.read_csv(path+'/Raw_Data_for_train_test.csv')
df.head()


# In[6]:


df.columns[df.isnull().any()]


# In[7]:


# Fill numeric rows with the median
for label, content in df.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Fill missing numeric values with median since it's more robust than the mean
            df[label] = content.fillna(content.median())
            
df.columns[df.isnull().any()]


# In[8]:


df.info()


# In[9]:


# Turn categorical variables into numbers
for label, content in df.items():
    # Check columns which aren't numeric
    if not pd.api.types.is_numeric_dtype(content):
        # print the columns that are objectt type 
        print(label)
        df[label] = pd.Categorical(content).codes+1


# In[10]:


# Cleaned data
df.head()


# In[11]:


# There's no need of Target_D column. As we are taking TARGET_B as our target variable. So we can drop this
df = df.drop('TARGET_D', axis=1)
df.head()


# In[12]:


# input features
x = df.drop('TARGET_B', axis=1)

# Target variable
y = df['TARGET_B']

x.head()


# In[13]:


y.head()


# In[14]:


# Import standard scaler
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

# apply scaler
x = ss.fit_transform(x)
x


# Modelling 
# We'll use following models and then evaluate them to find which model works well:
# 
# 1.KNN
# 2.Random Forest
# 3.XGBoost Classifier

# In[15]:


##KNN
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# define and configure the model
model = KNeighborsClassifier()

# fit the model
model.fit(xtrain, ytrain)

# evaluate the model
preds = model.predict(xtest)
accuracy_score(ytest, preds)


# In[16]:


## Random Forest
from sklearn.ensemble import RandomForestClassifier

# define and configure the model
model = RandomForestClassifier()

# fit the model
model.fit(xtrain, ytrain)

# evaluate the model
preds = model.predict(xtest)
accuracy_score(ytest, preds)


# In[17]:


## XGBOOST
from xgboost import XGBClassifier

# define and configure the model
model = XGBClassifier()

# fit the model
model.fit(xtrain, ytrain)

# evaluate the model
preds = model.predict(xtest)
accuracy_score(ytest, preds)


# ## We can see Random forest perfomed best. So let's perform hyperperameter tuning for Random forest

# In[18]:


import numpy as np
from sklearn.model_selection import RandomizedSearchCV

# different randomforestregressor hyperperameters
rf_grid = {'n_estimators' : np.arange(10, 100, 10),
           'max_depth': [None, 3, 5, 10],
           'min_samples_split' : np.arange(2, 20, 2),
           'min_samples_leaf': np.arange(1, 20, 2),
            'max_features' : [0.5, 1, 'sqrt', 'auto']}

# instentiate randomizedsearchcv model
rs_model= RandomizedSearchCV(RandomForestClassifier(n_jobs = -1, 
                                                  random_state=42),
                                                  param_distributions = rf_grid,
                                                  n_iter = 90,
                                                  cv=5,
                                                  verbose=True)

rs_model.fit(xtrain, ytrain)


# In[19]:


rs_model.best_params_


# ### We got the best parameters for our model. Now Let's create an ideal model that have these as it's parameters.

# In[20]:


ideal_model = RandomForestClassifier(n_estimators= 80,
                                     min_samples_split = 2,
                                     min_samples_leaf = 5,
                                     max_features = 'auto',
                                     max_depth = 10)

# fit the model
ideal_model.fit(xtrain, ytrain)

# evaluate the model
preds = ideal_model.predict(xtest)
accuracy_score(ytest, preds)


# In[21]:


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = ideal_model.predict_proba(xtest)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(ytest, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Now since we have a good model to predict. Let's Predict wheather a person donates or not for our Test data
# 

# In[22]:


test_df = pd.read_csv(path+'/Predict_donor.csv')
test_df.head()


# In[23]:


# Fill numeric rows with the median
for label, content in test_df.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Fill missing numeric values with median since it's more robust than the mean
            test_df[label] = content.fillna(content.median())


# In[24]:


# Turn categorical variables into numbers
for label, content in test_df.items():
    # Check columns which aren't numeric
    if not pd.api.types.is_numeric_dtype(content):
        # print the columns that are object type 
        print(label)
        test_df[label] = pd.Categorical(content).codes+1


# In[25]:


Target = ideal_model.predict(test_df)
Target


# In[26]:


PREDICTED_df = pd.DataFrame()
PREDICTED_df['TARGET_B'] = Target
PREDICTED_df['CONTROL_NUMBER'] = test_df['CONTROL_NUMBER']
PREDICTED_df.head()


# In[ ]:





# In[ ]:




