#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


path='E:\\DataScience\\MachineLearning\\Wine_Quality_Data'


# In[3]:


import os
os.listdir(path)


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import plotly.express as px
from wordcloud import WordCloud
from scipy import signal
import scipy
#to supress warning
import warnings
warnings.filterwarnings('ignore')


#to make shell more intractive
from IPython.display import display

# setting up the chart size and background
plt.rcParams['figure.figsize'] = (16, 8)
plt.style.use('fivethirtyeight')


# In[5]:


df = pd.read_csv(path+"\\WineQT.csv")


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


#Not of use basically it tell type of alcohol
df.drop('Id',axis=1,inplace=True)


# In[9]:


df.columns


# In[10]:


df.shape


# In[11]:


df.info()


# In[12]:


df.corr().style.background_gradient(cmap = 'rocket_r')


# In[13]:


df.describe().T.style.background_gradient(cmap = 'rocket_r')


# In[14]:


df.duplicated().sum()


# In[15]:


df.isnull().sum()


# In[16]:


#center of tendency
sns.distplot(df['fixed acidity'])


# In[17]:


sns.distplot


# In[ ]:





# In[18]:


def mix_plot(feature):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 3, 1)
    feature.plot(kind = 'hist')
    plt.title(f'{feature.name} histogram plot')
    #mean = feature.describe().mean()
    plt.subplot(1, 3, 2)
    mu, sigma = scipy.stats.norm.fit(feature)
    sns.distplot(feature) 
    #plt.legend({'--': mu, 'sigma': sigma})
    plt.axvline(mu, linestyle = '--', color = 'green', )
    plt.axvline(sigma, linestyle = '--', color = 'red')
    plt.title(f'{feature.name} distribution plot')
    plt.subplot(1, 3, 3)
    sns.boxplot(feature)
    plt.title(f'{feature.name} box plot')
    plt.show()


# In[19]:


for i in df.columns:
    mix_plot(df[i])


# In[20]:


df.groupby('quality').count()


# In[21]:


sns.pairplot(df,hue='quality')


# In[22]:


sns.heatmap(df.corr(),cbar=True,square=True,fmt='.2f',annot=True,annot_kws={'size':10},cmap='Pastel2')
sns.set_style('darkgrid')


# In[23]:


sns.scatterplot(data=df, x='fixed acidity', y='density',hue='quality')


# In[24]:


sns.scatterplot(data=df, x='volatile acidity', y='alcohol',hue='quality')


# In[25]:


## Normalize...
df["quality"].value_counts(normalize = True)


# In[26]:


# Independent variable and dependent variable
#Independent 
X = df.loc[:, df.columns != 'quality']
#dependent
y = df[['quality']]


# In[ ]:





# In[27]:


#Handel imbalace data set..
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


# In[28]:


y.value_counts()


# In[29]:


#split data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[30]:


print("Shape X_train :",X_train.shape)
print("Shape y_train :",y_train.shape)
print("Shape X_test  :",X_test.shape)
print("Shape y_test  :",y_test.shape)


# In[31]:


from sklearn.preprocessing import StandardScaler
Scaler =StandardScaler()
X =Scaler.fit_transform(X)


# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
import xgboost 
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold ,StratifiedKFold
from sklearn import metrics
from sklearn.pipeline import Pipeline


# In[33]:


pipe_LR=Pipeline([('scaler1',StandardScaler()),
                 ('LR',LogisticRegression(random_state=2))])
pipe_Ada=Pipeline([('scaler2',StandardScaler()),
                    ('Ada',AdaBoostClassifier(learning_rate=0.1,random_state=2))])
pipe_DT=Pipeline([('scaler3',StandardScaler()),
                  ('DTR',DecisionTreeClassifier())])
pipe_RF=Pipeline([('scaler4',StandardScaler()),
                  ('RFR',RandomForestClassifier())])
pipe_Knn=Pipeline([('scaler5',StandardScaler()),
                   ('Knn',KNeighborsClassifier())])
pipe_Xgb =Pipeline([('scaler5',StandardScaler()),
                   ('Xgboost',XGBClassifier(learning_rate=0.1,random_state=5))])


# In[34]:


pipeline=[pipe_LR,pipe_Ada,pipe_DT,pipe_RF,pipe_Knn,pipe_Xgb]
pipe_dict ={0:'Lr',1:'Ada',2:'DT',3:'RF',4:'Knn',5:'Xgb'}


# In[35]:


pipe_dict={0:'LogisticRegression',1:'AdaBoostClassifier',2:'DecisionTreeClassifier',3:'RandomForestClassifier'
           ,4:'KNeighborsClassifier',5:'XGBClassifier'}


# In[36]:


for pipe in pipeline:
  pipe.fit(X_train,y_train)


# In[37]:


for i,models in enumerate(pipeline):
  print("{} Accuracy : {}".format(pipe_dict[i],models.score(X_test,y_test)))


# In[38]:


model_XGB =XGBClassifier(learning_rate=0.1,random_state=5)
model_XGB.fit(X_train,y_train)
y_pred =model_XGB.predict(X_test)


# In[39]:


print('Accuracy_Score :',metrics.accuracy_score(y_test,y_pred))
print('Classification_report:\n',metrics.classification_report(y_test,y_pred))
print('Confusion_mat:\n',metrics.confusion_matrix(y_test,y_pred))


# In[ ]:




