#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries
~ Author : Saurabh Kumar
~ Date : 06-Jan-23
# In[2]:


import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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


# In[3]:


pwd


# In[4]:


import os
os.listdir()


# In[5]:


path ='E:\\DataScience\\MachineLearning\\Sonar vs Mine Prediction logistic regression'+'\\SonarData.csv'


# # Data Collection & Data processing

# In[6]:


#Loading the dataset to a pandas Dataframe
sonar_data = pd.read_csv(path,header=None)


# In[7]:


sonar_data.head()


# In[8]:


#Number of rows and columns
sonar_data.shape


# In[9]:


#decription --> statistical measures of the data
sonar_data.describe()


# In[10]:


sonar_data.columns


# In[11]:


sonar_data[60].value_counts()


# ### Correlation heatmap

# In[12]:


sonar_data.corr().style.background_gradient(cmap = 'rocket_r')


# In[13]:


sonar_data.describe().T.style.background_gradient(cmap = 'rocket_r')


# ### Barchart by rock and mine

# In[14]:


sonar_data.groupby(60)[60].count().plot.bar();


# In[15]:


col=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
     17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
     34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
     51, 52, 53, 54, 55, 56, 57, 58, 59]


# ### Boxplot by features and rock & mine

# In[17]:


fig = plt.figure(figsize=(20,15))

for i in range(len(col)):
    plt.subplot(6,10,i+1)
    plt.title(col[i])
    sns.boxplot(data=sonar_data,y=sonar_data[col[i]],x=sonar_data[60])

plt.tight_layout()   
plt.show()


# ### Histplot by features and rock & mine

# In[18]:


fig = plt.figure(figsize=(20,30))

for i in range(len(col)):
    plt.subplot(10,6,i+1)
    plt.title(col[i])
    sns.histplot(data=sonar_data,x=sonar_data[col[i]],hue=60)

plt.tight_layout()


# In[20]:


sonar_data.groupby(60).mean().style.background_gradient(cmap = 'rocket_r')


# ## Evaluate Algorithms
* Split-out validation dataset
* Test options and evaluation metric
* Spot Check Algorithms
* Compare Algorithms
# In[33]:


import operator 
## Preprocessing 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (cross_val_score, KFold,
                                     StratifiedKFold, train_test_split,
                                     GridSearchCV)
from sklearn.metrics import confusion_matrix, classification_report

## Piprline 
from sklearn.pipeline import make_pipeline 

### Linear Estimatiors
from sklearn.linear_model import LogisticRegression, SGDClassifier
### non Linear Estimatiors 
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier 
### Ensemble Estimatiors 

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier) 
### Metrics
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             classification_report,RocCurveDisplay)


# In[34]:


X = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]
'''
I split our data into Training and testing sets,
I specified my test size to 15% of the dataset 
and last, I chose a random state of 101
'''
X_train, X_test, y_train, y_test =train_test_split(
                            X,y,test_size=0.15, random_state = 101)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[35]:


np.random.seed(101)
models = {
    'LogisticRegression': LogisticRegression(),
    'SGDClassifier': SGDClassifier(),
    'LinearSVC': LinearSVC(),
    'SVC': SVC(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier':AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
}

Skfold = StratifiedKFold(n_splits = 13)
metrics = ['accuracy']


# In[36]:


np.random.seed(101)
score ={}
def mod(model):
    for k,v in model.items():
        v.fit(X_train, y_train)
        score[k] = np.round(cross_val_score(estimator= v,
                                            X= X_test,y= y_test, cv= Skfold, scoring = 'accuracy').mean(), 4)
    best = max(score.items(), key=operator.itemgetter(1))[0]
    print(f'Best Estimator : {best} with score = {100*score[best]:.2f}')
            
    return score
mod(models)


# In[37]:


# compare algorithms
results=mod(models)
estimators = list(results.keys())
score = list(results.values())


fig =  plt.figure(figsize= (10,5), dpi = 150)
plt.bar(range(len(results)), score, tick_label = estimators,
        color=(1, 0.1, 0.1, 0.7),  edgecolor='blue')
plt.xticks(rotation=45)
plt.xlabel('Estimators')
plt.ylabel('Accuracy')
plt.title('Comparison of Estimators on Accuracy');


# In[ ]:




