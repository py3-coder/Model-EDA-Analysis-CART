#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


path ='E:\\DataScience\\MachineLearning\\Ukraine_Russia_War_Analysis'
import os 
os.listdir(path)


# In[3]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots


# In[4]:


df_equip = pd.read_csv(path+"\\russia_losses_equipment.csv")


# In[5]:


df_per = pd.read_csv(path+"\\russia_losses_personnel.csv")


# In[6]:


df_equip.head()


# In[7]:


df_equip.tail()


# In[8]:


df_per.head()


# In[9]:


df_per.tail()


# In[10]:


x, y = df_per['date'],df_per['personnel']
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y,
                    mode='lines+markers',
                    name='lines+markers'))

fig.show()


# In[11]:


x = df_equip['date']
y0 = df_equip['aircraft']
y1 = df_equip['helicopter']
y2 = df_equip['anti-aircraft warfare']
y3 = df_equip['drone']

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y0,
                    mode='lines+markers',
                    name='Aircraft'))
fig.add_trace(go.Scatter(x=x, y=y1,
                    mode='lines+markers',
                    name='Helicopter'))
fig.add_trace(go.Scatter(x=x, y=y2,
                    mode='lines+markers',
                    name='Anti-aircraft warfare'))
fig.add_trace(go.Scatter(x=x, y=y3,
                    mode='lines+markers',
                    name='Drone'))
fig.update_layout(legend_orientation="h",
                  legend=dict(x=0, y=1, traceorder="normal"),
                  title="Weapons: Air",
                  xaxis_title="Date",
                  yaxis_title="Weapons ",
                  margin=dict(l=0, r=0, t=30, b=0))
fig.show()


# In[12]:


x  = df_equip['date']
y0 =df_equip['military auto']
y1 =df_equip['APC']
y2 =df_equip['fuel tank']
y3 =df_equip['tank']

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y0,
                    mode='lines+markers',
                    name='Military auto'))
fig.add_trace(go.Scatter(x=x, y=y1,
                    mode='lines+markers',
                    name='APC'))
fig.add_trace(go.Scatter(x=x, y=y2,
                    mode='lines+markers',
                    name='Fuel tank'))
fig.add_trace(go.Scatter(x=x, y=y3,
                    mode='lines+markers',
                    name='Tank'))
fig.update_layout(legend_orientation="h",
                  legend=dict(x=0, y=1, traceorder="normal"),
                  title="Weapons: Ground",
                  xaxis_title="Date",
                  yaxis_title="Weapons",
                  margin=dict(l=0, r=0, t=30, b=0))
fig.show()


# In[13]:


col = ['aircraft', 'helicopter', 'tank', 'APC','field artillery', 'MRL', 'military auto', 'fuel tank', 
       'drone','naval ship', 'anti-aircraft warfare', 'special equipment','mobile SRBM system']


# In[14]:


fig = go.Figure()
for i in col:
    fig.add_trace(go.Scatter(x=df_equip['date'], y=df_equip[i], mode='lines',name=i,))
fig.show()


# In[ ]:




