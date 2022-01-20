#!/usr/bin/env python
# coding: utf-8

# # Detection of Breast Cancer using ML
# @Author : Saurabh 
# @date   : 10 may 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[2]:


pwd


# In[3]:


path ='E:\\DataScience\\MachineLearning\\Breast-cancer-detection-using-ML'


# In[4]:


import os
os.listdir(path)


# In[5]:


df=pd.read_csv(path+"\\data.csv")


# In[6]:


df.head(7)


# In[7]:


df.shape


# In[8]:


#Counting number of empty columns
df.isna().sum()


# In[10]:


#drop all columns with missing values
df=df.dropna(axis=1)


# In[11]:


df.shape


# In[12]:


df['diagnosis'].value_counts()


# In[13]:


plt.figure(figsize=(9,9))
sns.countplot(df['diagnosis'],label='count')


# In[14]:


df.dtypes


# In[15]:


df.info()


# In[16]:


df.describe()


# In[17]:


#Encoding the categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()
labelencoder_Y.fit_transform(df.iloc[:,1].values)
df.iloc[:,1].values


# In[18]:


df.iloc[:,1]=labelencoder_Y.fit_transform(df.iloc[:,1].values)
df.iloc[:,1]


# In[19]:


#Creating pair plot
sns.pairplot(df.iloc[:,1:6], hue='diagnosis')


# In[20]:


df.head(5)


# In[21]:


#getting correlation of columns
df.iloc[:,1:12].corr()


# In[22]:


plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:12].corr(), annot=True, fmt='.0%')


# In[23]:


plt.figure(figsize=(9,9))
sns.distplot(df.iloc[:,1:12].corr())


# In[24]:


#Splitting data into independent X and Y
X=df.iloc[:,2:31].values
Y=df.iloc[:,1].values
type(X)


# In[25]:


#splitting data into 75%training and 25%testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.25,random_state=0)


# In[26]:


#Scaling the data
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

X_train


# In[27]:


#function for model
def models(X_train, Y_train):
    
    from sklearn.linear_model import LogisticRegression 
    log=LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)
    
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(criterion='entropy',random_state=0)
    tree.fit(X_train,Y_train)
    
    from sklearn.ensemble import RandomForestClassifier
    forest=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
    forest.fit(X_train,Y_train)
    
    print('[0]LogisticRegression Training Accuracy:',log.score(X_train,Y_train))
    print('[1]DecisionTree Training Accuracy:',tree.score(X_train,Y_train))
    print('[2]RandomForestClassifier Training Accuracy:',forest.score(X_train,Y_train))
    
    return log,tree,forest


# In[28]:


model=models(X_train,Y_train)


# In[29]:


#testing model accuracy on confusion matrix
from sklearn.metrics import confusion_matrix
for i in range(len(model) ):
    print('Model', i)
    cm= confusion_matrix(Y_test, model[i].predict(X_test))
    TP=cm[0][0]
    TN=cm[1][1]
    FN=cm[1][0]
    FP=cm[0][1]
    print(cm)
    print('Testing accuracy = ',(TP+TN)/(TP+TN+FN+FP))
    print()


# In[30]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model) ):
    print('Model', i)
    print(classification_report(Y_test,model[0].predict(X_test)))
    print(accuracy_score(Y_test, model[0].predict(X_test)))


# In[31]:


#prediction of Random Forest Classifier Model
pred=model[2].predict(X_test)
print(pred)
print()
print(Y_test)


# In[ ]:




