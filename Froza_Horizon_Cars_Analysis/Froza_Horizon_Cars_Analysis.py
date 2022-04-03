#!/usr/bin/env python
# coding: utf-8

# ## Author : Saurabh

# In[1]:


pwd


# In[31]:


import os
path ='E:\\DataScience\\MachineLearning\\Froza_Horizon_Cars_Data'
os.listdir(path)


# In[3]:


import pandas as pd
import math
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from IPython.display import Image, HTML


# In[4]:


df = pd.read_csv(path+"\\Froza_Horizon_Cars.csv")
def path_to_image_html(path):
    return '<img src="'+ path + '" style=max-height:124px;"/>'
HTML(df.to_html(escape=False ,formatters=dict(Car_Image=path_to_image_html)))


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


df.describe().T.round(2).style.background_gradient(cmap = 'Oranges')


# In[8]:


df['Drive_Type'].value_counts()


# In[9]:


df['Model'].value_counts()


# In[10]:


df['Model'].unique()


# In[11]:


df['Name'].unique()


# In[12]:


df['car_source'].unique()


# In[13]:


df['stock_specs'].unique()


# In[14]:


df.columns


# In[15]:


df['launch'].unique()


# In[16]:


df['Offroad'].unique()


# In[17]:


df.sort_values(['Model','Horse_Power'],ascending=False).groupby('Horse_Power').head(10)


# In[18]:


print(df.groupby('Model')['Horse_Power'].max())


# In[19]:


df['braking'].unique()


# In[20]:


df['Stock_Rating'].unique()


# In[21]:


df1 = df[['In_Game_Price','handling','acceleration','speed']]
df1


# In[22]:


(df1 == 'info_not_found').sum(axis=0)


# In[23]:


df1 = df1.replace(to_replace ='info_not_found',
                 value ='0')
(df1 == 'info_not_found').sum(axis=0)


# In[24]:


df1['In_Game_Price'] = df1['In_Game_Price'].str.replace(",","")
df1 = df1.astype({'In_Game_Price':float,'handling':float,'acceleration':float,'speed':float,})
df1.head()


# In[25]:


#df1.sort_values(by=['In_Game_Price'],inplace=True,ascending=False)
df1.sort_values(by=['speed'],inplace=True,ascending=False)
df1.sort_values(by=['acceleration'],inplace=True,ascending=False)
df1.sort_values(by=['handling'],inplace=True,ascending=False)


# In[26]:


top_cars = df1.head(10).index
list(top_cars)


# In[27]:


df2 = df.iloc[top_cars]
def path_to_image_html(path):
    return '<img src="'+ path + '" style=max-height:124px;"/>'
HTML(df2.to_html(escape=False ,formatters=dict(Car_Image=path_to_image_html)))


# In[28]:


df1.sort_values(by=['handling'],inplace=True,ascending=False)
df1.sort_values(by=['acceleration'],inplace=True,ascending=False)
df1.sort_values(by=['speed'],inplace=True,ascending=False)
df1


# In[29]:


top_cars = df1.head(10).index
list(top_cars)


# In[30]:


df2 = df.iloc[top_cars]
def path_to_image_html(path):
    return '<img src="'+ path + '" style=max-height:124px;"/>'
HTML(df2.to_html(escape=False ,formatters=dict(Car_Image=path_to_image_html)))


# In[ ]:




