#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


path ='E:\\DataScience\\MachineLearning\\Snake_Date'


# In[3]:


import os
os.listdir(path)


# In[4]:


#Importing Some Library
import os
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style('darkgrid')
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import signal
from wordcloud import WordCloud, STOPWORDS
import plotly.io as pio
pio.templates.default = "plotly_dark"

#to supress warning
import warnings
warnings.filterwarnings('ignore')

#to make shell more intractive
from IPython.display import display
from IPython.display import Image

# setting up the chart size and background
plt.rcParams['figure.figsize'] = (16, 8)
plt.style.use('fivethirtyeight')


# In[5]:


#read csv file
df = pd.read_csv(path+"\\ConservationStatusofWorldSnakes.csv")


# In[6]:


#data 
df


# In[7]:


#Diff. Family-Name of Snake available in this data 
df['Family'].unique()


# In[8]:


#No. of Diff. Family of Snake available in this data
df['Family'].nunique()


# In[9]:


fig = px.pie(df, values=df['Family'].value_counts(), names=df['Family'].value_counts().index
             , title="Families of Snakes")
fig.show()


# In[10]:


print(df['Population trend'].nunique())
print(df['Population trend'].unique())
print(df['Population trend'].value_counts())


# In[11]:


fig = px.bar(df,x=df['Population trend'].value_counts().index,y=df['Population trend'].value_counts(),
             color=df['Population trend'].value_counts().index,title="Population trend ")
fig.update_xaxes(title_text='Rate')
fig.update_yaxes(title_text='Count of Numbers')
fig.show()


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


print(df.isnull().sum())
sns.heatmap(df.isnull())


# In[15]:


#replacing the NAN of Name-not-found in Colum - Common names(s)
df['Common name(s)'].fillna('Name-not-found',inplace=True)


# In[16]:


print(df.isnull().sum())


# In[17]:


#replacing the NAN with unkown in Column - Population trend
df['Population trend'].fillna('unknown',inplace=True)


# In[18]:


fig = px.bar(df,x=df['Population trend'].value_counts().index,y=df['Population trend'].value_counts(),
             color=df['Population trend'].value_counts().index,title="Population trend ")
fig.update_xaxes(title_text='Rate')
fig.update_yaxes(title_text='Count of Numbers')
fig.show()


# In[19]:


text =" Snake Common Names ".join(cat for cat in df['Common name(s)'])
stop_words = list(STOPWORDS) + ["NaN","Name-not-found","found","Name"]
wordcloud = WordCloud(width=2000, height=1500, stopwords=stop_words, background_color='black', 
                      colormap='rocket_r', collocations=False, random_state=2022).generate(text)
plt.figure(figsize=(25,20))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

Red List Status 
1. EX - Extint
2. CR - Critically Endangered
3. EN - Endangered
4. VU - Vulnerable 
5. NT - Near Threatened
# In[20]:


df['Red List status'].unique()


# In[21]:


df['Red List status'].value_counts()


# In[22]:


## all data of those snake who is Extint 
df.loc[df['Red List status'] == 'EX']


# In[23]:


df.loc[df['Red List status'] == 'CR']


# In[24]:


## Family wise max snake scientific name ::
print(df.groupby('Family')['Scientific name'].max())


# In[25]:


## Family wise min snake scientific name ::
print(df.groupby('Family')['Scientific name'].min())


# In[26]:


## Family wise Count of  snake (scientific name) available ::
print(df.groupby('Family')['Scientific name'].count())


# In[27]:


fig = px.bar(df,x=df.groupby('Family')['Scientific name'].count( ).index,y=df.groupby('Family')['Scientific name'].count(),
             color=df.groupby('Family')['Scientific name'].count().index,title="Family wise Count of Snake")
fig.update_xaxes(title_text='Family-names of Snake')
fig.update_yaxes(title_text='Count of Snakes in each family')
fig.show()


# In[28]:


fig = px.sunburst(df,path=['Family','Red List status'],
                 color_discrete_sequence =px.colors.qualitative.Dark24,
                 title = 'Family ~ Red List Status')
fig.show()


# In[29]:


fig = px.sunburst(df,path=['Red List status','Population trend'],
                 color_discrete_sequence =px.colors.qualitative.Dark24,
                 title = 'Red List status ~ Population Trend')
fig.show()


# In[30]:


fig = px.sunburst(df,path=['Family','Population trend'],
                 color_discrete_sequence =px.colors.qualitative.Dark24,
                 title = 'Family ~ Population Trend')
fig.show()


# In[31]:


groups = df['Family'].dropna(False)
plt.subplots(figsize=(20,10))
wordcloud = WordCloud(background_color = 'Black',
                     width = 512,
                     height = 384,).generate(' '.join(groups))
plt.axis('off')
plt.imshow(wordcloud)
plt.title('Popular Families of Snake', 
        fontdict={'family': 'serif',
        'color':  'red',
        'weight': 'bold',
        'size': 30,})
plt.show()


# In[32]:


text =" Snake Common Names ".join(cat for cat in df['Common name(s)'])
stop_words = list(STOPWORDS) + ["NaN","Name-not-found","found","Name"]
wordcloud = WordCloud(width=2000, height=1500, stopwords=stop_words, background_color='black', 
                      colormap='rocket_r', collocations=False, random_state=2022).generate(text)
plt.figure(figsize=(25,20))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[ ]:




