#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


import os 
os.listdir('E:\\DataScience\\MachineLearning\\EDA_Terror_Data')


# In[3]:


path ='E:\\DataScience\\MachineLearning\\EDA_Terror_Data'


# In[6]:


get_ipython().system('pip install plotly==4.5')


# In[9]:


get_ipython().system('pip install wordcloud')


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import plotly.express as px
from wordcloud import WordCloud
from scipy import signal

#to supress warning
import warnings
warnings.filterwarnings('ignore')


#to make shell more intractive
from IPython.display import display

# setting up the chart size and background
plt.rcParams['figure.figsize'] = (16, 8)
plt.style.use('fivethirtyeight')


# ## **Data Analyzing**

# In[11]:


df =pd.read_csv(path+"\\globalterrorismdb_data.csv",encoding='latin1')


# In[12]:


#head
df.head()


# In[13]:


#shape
df.shape


# In[14]:


#info
df.info()


# In[15]:


#describe
df.describe()


# In[16]:


#null return total 
df.isnull().sum().sum()


# In[17]:


#checking the null percolums
df.isnull().sum()


# In[18]:


#adding kill and nwound to get casualities
df['casualities'] = df['nkill'] + df['nwound']


# In[19]:


#just to display columns as row-wise to look all columns at once
columns = df.columns.tolist()         
print(columns)


# In[20]:


df.head()


# In[21]:


df_terror =df[['iyear','country_txt','region_txt','city','latitude','longitude','success','suicide','attacktype1_txt','targtype1_txt','nkill','target1','dbsource','casualities']]


# In[22]:


df_terror.head()


# In[23]:


print('Country having highest terrorist attack:',df_terror['country_txt'].value_counts().index[0])
print('Year having Maximum number of attacks :',df_terror['iyear'].value_counts().idxmax())
print('Region having highest number of attack:',df_terror['region_txt'].value_counts().index[0])
print('Maximum people killed in an attack are:',df['nkill'].max(),'that took place in',df.loc[df['nkill'].idxmax()].country_txt)


# In[24]:


df['attacktype1_txt'].value_counts()


# In[25]:


df['targtype1_txt'].value_counts()


# In[26]:


df['country_txt'].value_counts()


# ### Countries Under Threat

# In[27]:


plt.figure(figsize = (18,6))
p = df['country_txt'].value_counts().index
sns.countplot(df['country_txt'], 
              order = p[:26],
              palette = sns.color_palette("flare"))
plt.xlabel('Country')
plt.ylabel('Death Count')
plt.title('Top 25 Countries Under Threat')
plt.xticks(rotation = 90)
plt.show()


# ### Comparing No. of Attacks with Killings for each country

# In[28]:



attacks = df.country_txt.value_counts()[:20].to_frame()
attacks.columns = ['Attacks']
kills = df.groupby(['country_txt'])['nkill'].sum().sort_values(ascending =False).to_frame()
attacks.merge(kills, how = 'left' , left_index = True, right_index = True ).plot.bar(width = 0.8 , color = sns.color_palette('flare',2))
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.ylabel("Count" ,fontsize = 15)
plt.xlabel("Country",fontsize = 15)
plt.show()


# ### Forms of Attack

# In[29]:


plt.figure(figsize = (13,7))
sns.countplot(df['attacktype1_txt'], 
              order = df['attacktype1_txt'].value_counts().index,
              palette = sns.color_palette("flare"))
plt.xlabel('Weapons Used')
plt.ylabel('Resulting Death Count')
plt.title('Forms of Attack')
plt.xticks(rotation = 90)
plt.show()


# ### Type of Attacks in ecah year

# In[30]:


pd.crosstab(df.iyear,df.attacktype1_txt).plot(kind='area',figsize=(15,6))
plt.title('Type of Attacks in ecah year')
plt.ylabel('Type of Attacks')
plt.show()


# ### Terrorist Activities by Region in each Year

# In[31]:


pd.crosstab(df.iyear, df.region_txt).plot(kind='area',figsize=(15,6))
plt.title('Terrorist Activities by Region in each Year')
plt.ylabel('Number of Attacks')
plt.show()


# ### Region having Maximum number of attacks

# In[32]:


plt.figure(figsize = (18,6))
p = df['region_txt'].value_counts().index
sns.countplot(df['region_txt'], 
              order = p[:11],
              palette = sns.color_palette("flare"))
plt.xlabel('Region')
plt.ylabel('Attacks')
plt.title('Region having Maximum number of attacks')  
plt.xticks(rotation = 90)
plt.show()


# In[33]:


fig = px.bar(df, x='iyear', y='nkill', color='gname',
             labels = {'iyear':'Year', 'nkill':'Number of Deaths','country_txt':'Country', 'nwound':'Wounded', 'gname':'Extremist Group','region_txt':'Region'},
             title = 'Number of Dealths in a Year and the Responsible Extremists',
             hover_data = ['nwound','region_txt'])
fig.show()


# In[34]:


fig = px.sunburst(df, values='casualities',
                 path=['region_txt','attacktype1_txt'],
                 color_discrete_sequence =px.colors.qualitative.Dark24,
                 title = 'Sunburst Chart: Grouping the Type of Attacks in Different Regions')
fig.show()


# In[35]:


plt.subplots(figsize=(15,6))
sns.countplot('iyear',data=df,palette='afmhot_r',edgecolor=sns.color_palette('dark',5))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()


# In[36]:


plt.subplots(figsize=(15,6))
sns.countplot('imonth',data=df,palette=sns.color_palette('Paired',10),edgecolor=sns.color_palette('dark',5))
plt.title('Number Of Terrorist Activities Each Month')
plt.show()


# In[37]:


cities = df.target1.dropna(False)
plt.subplots(figsize=(20,10))
wordcloud = WordCloud(background_color = 'Black',
                     width = 512,
                     height = 384,).generate(' '.join(cities))
plt.axis('off')
plt.imshow(wordcloud)
plt.title('Popular Targets', 
        fontdict={'family': 'serif',
        'color':  'red',
        'weight': 'bold',
        'size': 26,})
plt.show()


# In[13]:


groups = df['gname'].dropna(False)
plt.subplots(figsize=(20,10))
wordcloud = WordCloud(background_color = 'Black',
                     width = 512,
                     height = 384,).generate(' '.join(groups))
plt.axis('off')
plt.imshow(wordcloud)
plt.title('Popular Terrist Group', 
        fontdict={'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 26,})
plt.show()


# In[38]:


motive = df['motive'].dropna(False)
plt.subplots(figsize=(20,10))
wordcloud = WordCloud(background_color = 'Black',
                     width = 512,
                     height = 384,).generate(' '.join(motive))
plt.axis('off')
plt.imshow(wordcloud)
plt.title('Motive Behind Attacks', 
        fontdict={'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 26,})
plt.show()


# In[39]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
stopwords = set(STOPWORDS)
stopwords.update(["Unknown",'unknown','specific','motive','sources','noted','part','stated','posited','speculated','attack','attacks','suspected'])
motive = df.motive.dropna()
terror_mask = np.array(Image.open(path+"//terrorist-attack-and-war-.jpg"))
plt.subplots(figsize=(18,15))
wordcloud_fra = WordCloud(background_color = 'black',
                     width = 512,
                     height = 384,stopwords = stopwords, max_font_size=100, random_state=43, contour_color='firebrick',mask = terror_mask, max_words=20000).generate(' '.join(motive))
image_colors = ImageColorGenerator(terror_mask)

#wordcloud_fra.to_file(("city_m.png"))

plt.axis('off')
plt.imshow(wordcloud_fra)

plt.savefig("terror_word_cloud.png", format="png")
plt.title('Terrist', 
        fontdict={'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 26,})
plt.show()


# In[ ]:




