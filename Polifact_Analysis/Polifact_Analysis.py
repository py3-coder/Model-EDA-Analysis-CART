#!/usr/bin/env python
# coding: utf-8

# # Polifact_Analysis
# 

# ### @Author : Saurabh Kumar

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style('darkgrid')
import plotly.express as px
from scipy import signal
import warnings
warnings.filterwarnings("ignore")

#to make shell more intractive
from IPython.display import display
from IPython.display import Image

# setting up the chart size and background
plt.rcParams['figure.figsize'] = (16, 8)
plt.style.use('fivethirtyeight')


# In[2]:


pwd


# In[3]:


path ='E:\\DataScience\\MachineLearning\\Polotifact_Data'


# In[4]:


import os
from glob import glob
os.listdir(path)


# In[5]:


df = pd.read_csv(path+"\\politifact.csv")
df.head(5)


# In[6]:


pd.set_option('display.max_colwidth', 200) 
df.head(10)


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df.nunique(axis=0)


# In[11]:


df['source'].unique()


# In[12]:


df['veracity'].unique()


# In[13]:


df['source'].value_counts()


# In[14]:


df['veracity'].value_counts()


# In[15]:


# Veracity means : conformity with truth or fact
# Full Flop , Half Flip , No Flop does not give any clear meaning in sence of Veracity
# Half - True it contains 50 % True and 50 % False 

df_new = df[~df.isin(['Half-True','Full Flop','No Flip','Half Flip']).any(axis=1)]


# In[16]:


df_new.head(5)


# In[17]:


df_new['veracity'].value_counts()


# In[18]:


#df_new = df_new.replace({'False': 0 ,'Mostly False' : 0 , 'Pants on Fire!' : 0, 'Mostly True' : 1 , 'True' : 1 })


# In[20]:


df_new.head()


# In[21]:


df_new.shape


# In[22]:


Source_top_10 = df_new['source'].value_counts().nlargest(10)
Source_top_10


# In[23]:


df_10 = df_new[df_new.isin(['Barack Obama','Donald Trump','Bloggers','Hillary Clinton','Chain email','John McCain',
                    'Scott Walker','Mitt Romney','Rick Perry','Marco Rubio']).any(axis=1)] 
df_10.head()


# In[24]:


df_10['source'].value_counts()
sns.histplot(data=df_10,x='source',kde=True, hue=df_10['veracity'])


# In[25]:


fig = px.pie(df_10 , values =Source_top_10 , 
             names =['Barack Obama','Donald Trump','Bloggers','Hillary Clinton','Chain email','John McCain',
                    'Scott Walker','Mitt Romney','Rick Perry','Marco Rubio'],
             title = 'Top 10 Sources From where Statement is taken :',
             labels={'source'})
fig.show()


# In[26]:


# 0 - False statement , 1- True statement  .....
fig = px.sunburst(df_10, names=None, values=None, parents=None, path=['source','veracity'], 
                   color='veracity', color_continuous_scale=None, range_color=None, color_continuous_midpoint=None, 
                  color_discrete_sequence=None,
                 labels={'source','veracity'}, title= "Detailed_Analysis Chart : 0 - False Statement & 1 - True Statement")
fig.show()


# In[27]:


remove = "/web/20180705082623"
len(remove)


# In[28]:


df_new['link'] =df_new['link'].apply(lambda x: x[len(remove):])


# In[29]:


df_new.head()


# In[30]:


# extracting the date from the link column
# r() = first group to extract; you can use multiple ()
# ?P<column_name> = column name for convenience
# \d = digit
# {n} = number of digits to include
# . = wildcard
# + = greedy search
# ? = but not too greedy
df_new["date"] = df_new.link.str.extract(r'(\d{4}/.+?\d{2})')


# In[31]:


df_new.head()


# In[32]:


# change the date column to a datetime column for convenience
df_new.date = pd.to_datetime(df_new.date,infer_datetime_format=True)


# In[33]:


df_new.head()


# In[34]:


df_new.info()


# In[35]:


df_new.shape


# In[36]:


#link are not important factor since most of them are from same web sites
df_new = df_new.drop(['link'], axis=1)
df_new


# In[37]:


#new colums of year in data
df_new['year'] = df_new['date'].dt.year


# In[38]:


df_new.info()


# In[39]:


df_new.head(5)


# In[40]:


df_new['year'].value_counts()


# In[41]:


df_new['source'].value_counts()


# In[42]:


##Lets analise some more intresting facts from data 
#year wise statement given by sources are True or False with percentage


# # Barack Obama

# In[43]:


##Barack Obama  
df_Barack = df_new[df_new['source'] == 'Barack Obama']


# In[44]:


df_Barack.head(10)


# In[45]:


df_Barack.reset_index(inplace=True)


# In[46]:


df_Barack.drop('index', axis=1,inplace= True)
df_Barack.head()


# In[47]:


#statement per year by Barack Obama
sns.histplot(data=df_Barack,x='year',kde=True, hue=df_Barack['veracity'])


# In[48]:


#statement per year by Barack Obama
sns.histplot(data=df_Barack,x='year',kde=True, hue=df_Barack['veracity'],stat='density',element='poly')


# In[49]:


fig = px.sunburst(df_Barack, names=None, values=None, parents=None, path=['year','veracity'], 
                   color='veracity', color_continuous_scale=None, range_color=None, color_continuous_midpoint=None, 
                  color_discrete_sequence=None,
                 labels={'year','veracity'}, title= "Barack obama Analysis")
fig.show()


# In[50]:


df_Barack.replace({'False': 0 ,'Mostly False' : 0 , 'Pants on Fire!' : 0, 'Mostly True' : 1 , 'True' : 1 },inplace = True)


# In[51]:


df_Barack.tail(5)


# In[52]:


fig = px.sunburst(df_Barack, names=None, values=None, parents=None, path=['year','veracity'], 
                   color='veracity', color_continuous_scale=None, range_color=None, color_continuous_midpoint=None, 
                  color_discrete_sequence=None,
                 labels={'year','veracity'}, title= "Barack obama Analysis")
fig.show()


# In[53]:


sns.histplot(data=df_Barack,x='year',kde=True, hue=df_Barack['veracity'],stat='probability',element='poly')


# # Donald Trump

# In[54]:


df_Donald = df_new[df_new['source']=='Donald Trump']


# In[55]:


df_Donald.head(5)


# In[56]:


df_Donald.reset_index(inplace=True)
df_Donald.drop('index', axis=1,inplace= True)
df_Donald.head()


# In[57]:


#statement per year by Barack Obama
sns.histplot(data=df_Donald,x='year',kde=True, hue=df_Barack['veracity'])


# In[58]:


sns.histplot(data=df_Donald,x='year',kde=True, hue=df_Barack['veracity'],stat='density',element='poly')


# In[59]:


fig = px.sunburst(df_Donald, names=None, values=None, parents=None, path=['year','veracity'], 
                   color='veracity', color_continuous_scale=None, range_color=None, color_continuous_midpoint=None, 
                  color_discrete_sequence=None,
                 labels={'year','veracity'}, title= " Donald Analysis")
fig.show()


# In[60]:


df_Donald.replace({'False': 0 ,'Mostly False' : 0 , 'Pants on Fire!' : 0, 'Mostly True' : 1 , 'True' : 1 },inplace = True)


# In[61]:


df_Donald.head(20)


# In[62]:


count =df_Donald['veracity'].value_counts()
count


# In[63]:


fig = px.sunburst(df_Donald, names=None, values=None, parents=None, path=['year','veracity'], 
                   color='veracity', color_continuous_scale=None, range_color=None, color_continuous_midpoint=None, 
                  color_discrete_sequence=None,
                 labels={'year','veracity'}, title= "Barack obama Analysis")
fig.show()


# In[64]:


df_new.head(10)


# In[65]:


df_new.replace({'False': 0 ,'Mostly False' : 0 , 'Pants on Fire!' : 0, 'Mostly True' : 1 , 'True' : 1 },inplace = True)


# In[66]:


df_new.head(5)


# In[67]:


df_clean = df_new.iloc[:,:-2]


# In[68]:


#final file... 

df_clean


# In[70]:


df_clean.to_csv("Clean_File.csv")


# In[ ]:




