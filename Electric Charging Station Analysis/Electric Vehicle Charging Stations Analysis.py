#!/usr/bin/env python
# coding: utf-8

# ## @Author : Saurabh

# In[1]:


pwd


# In[2]:


path='E:\\DataScience\\MachineLearning\\Electric Vehicle Charging Stations_data'
import os 
os.listdir(path)


# In[3]:


dataset =path+"\\ev_stations_v1.csv"


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
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

#map and other location lib
import folium
from folium import Marker
from folium.plugins import MarkerCluster


# In[5]:


df =pd.read_csv(dataset)


# In[6]:


df.head(5)


# In[7]:


df.isnull().sum()


# In[8]:


#plot for null values
sns.set(rc={'figure.figsize':(12,10)})
sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap='Dark2_r')


# In[9]:


#percentage of null each columns
for col in df.columns:
    print('{} :NUll %  : {}'.format(col,df[col].isnull().sum()/df[col].count()))


# In[10]:


#shape
df.shape


# In[11]:


df.info()


# In[12]:


df.columns


# In[13]:


#selecting imp. columns
df_new =df.loc[:,['Fuel Type Code', 'Station Name', 'Street Address',
        'City', 'State', 'ZIP','Station Phone', 
        'Status Code','Groups With Access Code', 'Access Days Time', 'EV Network',
       'EV Network Web', 'Geocode Status', 'Latitude', 'Longitude',
       'Date Last Confirmed', 'ID', 'Updated At','Open Date','EV Connector Types', 'Country',
        'Access Code']]


# In[14]:


df_new.head()


# In[15]:


#plot for null values of df_new
sns.set(rc={'figure.figsize':(14,10)})
sns.heatmap(df_new.isnull(),cbar=False,yticklabels=False,cmap='Dark2_r')


# In[16]:


df_new.head()


# In[17]:


df_new['Groups With Access Code'].unique()


# In[18]:


df_new['Groups With Access Code'].value_counts()


# In[19]:


df_new['Country'].value_counts()


# In[20]:


df_new['State'].value_counts()


# In[21]:


df_new['Access Code'].value_counts()  


# In[22]:


fig =px.histogram(df_new, x = 'Access Code', nbins=30)
fig.show()


# In[23]:


top_10_city=df_new['City'].value_counts().nlargest(10)
top_10_city


# In[24]:


fig = px.bar(df_new,y=df_new['City'].value_counts().nlargest(10),x=['Los Angeles','San Diego','Atlanta',
                                'San Jose','Irvine','Austin','San Francisco','Kansas City','Seattle','Boston'],
            color=['Los Angeles','San Diego','Atlanta','San Jose',
                   'Irvine','Austin','San Francisco','Kansas City','Seattle','Boston'])
fig.show()


# In[25]:


df_new['State'].value_counts().nlargest(15)
sns.barplot(data= df_new,y=df_new['State'].value_counts().nlargest(15),x=['CA','NY','FL','TX','MA','WA','CO','GA',
                                                                          'MD','PA','VA','NC','IL','MO','OR'])


# In[26]:


df_new['Latitude']


# In[27]:


df_new['Longitude']


# In[28]:


# Create a map centered on Charlotte, North Carolina
"""us_map = folium.Map(location=[34.248319,-118.387971], tiles='openstreetmap', zoom_start=11)

for idx, row in df_new.iterrows():
    Marker(location=[row['Latitude'], row['Longitude']],
           popup=row['Street Address']).add_to(us_map)
us_map """

#to see full map of us 


# In[29]:


loc_df = df_new[df_new['Latitude'].notnull() & df_new['Longitude'].notnull()]
# State --- NC    and    City --  Charlotte
Charlotte_NC_loc_df = loc_df[(loc_df['State'] == 'NC') & (loc_df['City'] == 'Charlotte')]
Charlotte_NC_loc_df.head()


# ## Map

# In[30]:


# Create a map centered on Charlotte, North Carolina
charlotte_map = folium.Map(location=[35.227,-80.843], tiles='openstreetmap', zoom_start=11)

# Add points to the map
for idx, row in Charlotte_NC_loc_df.iterrows():
    Marker(location=[row['Latitude'], row['Longitude']],
           popup=row['Street Address']).add_to(charlotte_map)

# Display the map
charlotte_map


# In[31]:


#For  state == NC 
NC_loc_df = loc_df[(loc_df['State'] == 'NC')]
NC_loc_df.head()


# In[32]:


# Create a map centered on North Carolina
nc_map = folium.Map(location=[35.5908438,-79.7628341], tiles='openstreetmap', zoom_start=7)

# Add points to a marker cluster
mc = MarkerCluster()
for idx, row in NC_loc_df.iterrows():
    mc.add_child(Marker(location=[row['Latitude'], row['Longitude']],
                        popup=row['Street Address']))

# add the marker cluster to the map
nc_map.add_child(mc)

# Display the map
nc_map


# In[33]:


top_10_city


# In[34]:


## lets us see the map for Los Angeles
LA_loc_df = loc_df[(loc_df['City'] == 'Los Angeles')]
LA_loc_df.head()


# In[35]:


LA_loc_df.shape


# In[36]:


# Create a map centered on LA
LA_map = folium.Map(location=[34.052542,-118.448504], tiles='openstreetmap', zoom_start=7)

# Add points to a marker cluster
mc = MarkerCluster()
for idx, row in LA_loc_df.iterrows():
    mc.add_child(Marker(location=[row['Latitude'], row['Longitude']],
                        popup=row['Street Address']))

# add the marker cluster to the mapś
LA_map.add_child(mc)

# Display the map
LA_map


# In[ ]:




