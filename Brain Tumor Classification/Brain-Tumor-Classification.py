#!/usr/bin/env python
# coding: utf-8

# ## Brain Tumor Classification

# In[3]:


pwd  


# In[4]:


path='E:\\DataScience\\MachineLearning\\Brain_Tumor_Data'


# In[5]:


import os
os.listdir(path)


# In[6]:


#importing lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import plotly.express as px
from wordcloud import WordCloud
from scipy import signal

#to supress warning
import warnings
warnings.filterwarnings('ignore')


# setting up the chart size and background
plt.rcParams['figure.figsize'] = (16, 8)
plt.style.use('fivethirtyeight')


# In[7]:


df = pd.read_csv(path+"\\data.csv")


# In[8]:


df.head(10)


# In[9]:


df.shape


# In[10]:


df.info()


# In[11]:


df.columns


# In[12]:


df.describe()


# In[13]:


df['y'].value_counts()


# In[14]:


df.tail(5)


# In[15]:


df.drop('Unnamed: 0', axis=1,inplace=True)


# In[16]:


df.head()


# ## Binaries Target Column

# In[17]:


target =pd.get_dummies(df['y'],dummy_na=True)


# In[18]:


target


# In[19]:


target = target.iloc[:,1]


# In[20]:


target


# In[21]:


df = pd.concat([df,target],axis=1)


# In[22]:


df.head()


# In[23]:


df.drop('y',axis=1,inplace=True)


# In[24]:


df.head(3)


# ## Feature_Engineering

# In[25]:


#importing feauture_selection from sklearn
# stastistical fuction --- chi 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 ,f_classif


# In[26]:


# X - independent variable , y - dependent varibale
X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[27]:


X.shape


# In[28]:


X.head()


# In[29]:


# k=50 : selecting top=50 rows values  which are highly co-related to taget 
# using f_classif function
fs_f =SelectKBest(f_classif ,k=50)
X_f_classif = fs_f.fit_transform(X,y)


# In[30]:


#X_f_classif Selected
dfscores = pd.DataFrame(fs_f.scores_)
dfcolumns =pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#naming the dataframe columns
featureScores.columns = ['Specs','Score']
print(featureScores.nlargest(50,'Score')) 


# In[31]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
 
#calling fit function 
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers


# In[32]:


#plot graph of feature importances for better visualization
plt.figure(figsize=(18,25))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(50).plot(kind='barh')
plt.show()


# In[33]:


lst =df.columns
lst


# In[34]:


lst_f =featureScores.nlargest(50,'Score')
lst_f=lst_f['Specs']
lst_f = list(lst_f)
lst_f


# In[38]:


df_new = df[['M77836',
 'M83670','T96548','U17077','H06524','J02854','M97496',
 'T64297','U14631','M36634','M12272','Z49269.1','T51961','R61502','M26697','X73502','T67077',
 'H57136','H43887','R88575','T60155','M95787','Z49269.2','R08183','X12671','M22382','U37019','R71676','X12496','M80244',
 'T71025','M95936','T52362','H40095','X70326','M84526','Z31695','M63391','Z49269','T76971','R99208','M63603','L03840',
 'M76378.1','T51913','M76378.2','D63874','H65842','H09351','R46753','tumor']]


# In[39]:


df_new


# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


# In[41]:


## independent and dependent varible
X = df_new.iloc[:,:-1]
y = df_new.iloc[:,-1]


# In[42]:


# train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=0)


# In[43]:


##
pipe_LR=Pipeline([('scaler1',StandardScaler()),
                 ('LR',LogisticRegression())])
pipe_SVM =Pipeline([('scaler2',StandardScaler()),
                    ('SVR',SVC())])
pipe_DT=Pipeline([('scaler3',StandardScaler()),
                  ('DTR',DecisionTreeClassifier())])
pipe_RF=Pipeline([('scaler4',StandardScaler()),
                  ('RFR',RandomForestClassifier())])
pipe_Knn=Pipeline([('scaler4',StandardScaler()),
                   ('Knn',KNeighborsClassifier())])


# In[44]:


pipeline1=[pipe_LR,pipe_SVM,pipe_DT,pipe_RF,pipe_Knn]


# In[45]:


pipe_dict={0:'Logistic_Regression',1:'SVC',2:'Decesion_Tree_Classifier',3:'Random_Tree_classifier',4:'KNN_classifier'}


# In[46]:


for pipe in pipeline1:
  pipe.fit(X_train,y_train)


# In[47]:


from sklearn.metrics import accuracy_score


# In[48]:


for i,model in enumerate(pipeline1):
  print("{}_Accuracy: {}".format(pipe_dict[i],model.score(X_test,y_test)))


# In[ ]:




