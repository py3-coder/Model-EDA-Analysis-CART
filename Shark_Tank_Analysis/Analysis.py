#!/usr/bin/env python
# coding: utf-8

# ### Author : Saurabh

# # Analysis of Shark Tank Data

# In[7]:


#Poster of Tag 
Image(path+"\\Sharkimgtitle.jpg",height=420,width=1000)


# In[8]:


text =" Shark Tank India ".join(cat for cat in df['Episode Title'])
stop_words = list(STOPWORDS) + ["Ka", "Ki", "Ko","Karne","Wale","Aam","Aur","Ek"]
wordcloud = WordCloud(width=2000, height=1500, stopwords=stop_words, background_color='black', colormap='rocket_r', collocations=False, random_state=2022).generate(text)
plt.figure(figsize=(25,20))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[9]:


# Judges of Sharks Tank image:
Image(path+"\\Sharksimage.jpg")


# In[1]:


pwd


# In[2]:


path='E:\\DataScience\\MachineLearning\\Shark Tank Data Analysis'


# In[3]:


import os
os.listdir(path)


# In[4]:


import os
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style('darkgrid')
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import signal
from wordcloud import WordCloud, STOPWORDS
from babel.numbers import format_currency
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


# In[106]:


df=pd.read_csv(path+"\\Shark Tank India.csv")


# In[6]:


df.head(15).style.background_gradient(cmap = 'Oranges')


# In[10]:


df['Male Presenters'] =   df['Male Presenters'].astype(pd.Int32Dtype())
df['Female Presenters'] = df['Female Presenters'].astype(pd.Int32Dtype())
df['Started in'] =        df['Started in'].astype(pd.Int32Dtype())
df['Yearly Revenue'] =    df['Yearly Revenue'].astype(pd.Int32Dtype())
df['Monthly Sales'] =     df['Monthly Sales'].astype(pd.Int32Dtype())
df['Valuation Offered'] = df['Valuation Offered'].astype(pd.Int32Dtype())


# In[11]:


df.head(5)


# In[12]:


df.shape


# In[13]:


df.info()


# In[14]:


df.describe().T.round(2).style.background_gradient(cmap = 'Oranges')


# In[15]:


df.columns


# ## Some Facts of Shark Tank Data

# In[16]:


print("Shark Tank Episodes are Boradcasted on SonyLiv OTT Platform")
print("Season :",df['Season Number'].max())
print("Total number of Episodes in Covered:",df['Episode Number'].max())
print("Total number of Pitch  :",df['Pitch Number'].max())


# In[17]:


df['Industry'].value_counts()


# In[18]:


df['Industry'].value_counts().index


# In[19]:


fig = px.bar(df,x=df['Industry'].value_counts().index,y=df['Industry'].value_counts(),
             color=df['Industry'].value_counts().index,title="Types of Industry")
fig.update_xaxes(title_text='Industry')
fig.update_yaxes(title_text='No. of Startsups in those Sector')
fig.show()


# In[20]:


print("Total number of  Presenters :",df['Number of Presenters'].sum(),"\n")
print("Total number of Male Presenters :",df['Male Presenters'].sum(),"\n")
print("Total number of Female Presenters :",df['Female Presenters'].sum(),"\n")
print("Male Entrepreneurs % :",round(df['Male Presenters'].sum()/df['Number of Presenters'].sum()*100, 0),"\n")
print("Female Entrepreneurs % :",round(df['Female Presenters'].sum()/df['Number of Presenters'].sum()*100, 0),"\n")


# In[21]:


print("Total number of Couple  Presenters :",df['Couple Presenters'].sum(),"\n")
print("Couple Entrepreneurs % :",round(df['Couple Presenters'].sum()/df['Number of Presenters'].sum()*100, 0),"\n")


# In[22]:


print(df['Pitchers State'].value_counts())
fig = px.bar(df,x=df['Pitchers State'].value_counts().index,y=df['Pitchers State'].value_counts(),
             color=df['Pitchers State'].value_counts().index,title="Pitchers State")
fig.update_xaxes(title_text='State')
fig.update_yaxes(title_text='No. of Startups')
fig.show()


# In[23]:


fig = px.pie(df, values=df['Pitchers State'].value_counts(), names=df['Pitchers State'].value_counts().index
             , title='State % Share of Entrepreneurs:')
fig.show()


# In[24]:


#Average age type value count
df['Pitchers Average Age'].value_counts()


# In[25]:


# Most of the Avg age of Entrepreneurs is null 
df['Pitchers Average Age'].isnull().sum()


# In[26]:


## top 20 city from where the most pitches came..
df['Pitchers City'].value_counts().nlargest(20)


# In[42]:


fig = px.pie(df, values=df['Pitchers City'].value_counts().nlargest(15), names=df['Pitchers City'].value_counts().nlargest(15).index
             , title='Startup Distribution Top 15 Cities ')
fig.show()


# In[28]:


print(df.groupby('Startup Name')['Yearly Revenue'].max().nlargest(15))
fig = px.bar(df,x=df.groupby('Startup Name')['Yearly Revenue'].max().nlargest(15).index,y=df.groupby('Startup Name')['Yearly Revenue'].max().nlargest(15),
             color=df.groupby('Startup Name')['Yearly Revenue'].max().nlargest(15).index,title="Top 15 Startups which generates highest Revenue [in Lakhs]")
fig.update_xaxes(title_text='Startups')
fig.update_yaxes(title_text='Yearly Revenue (lakhs)')
fig.show()


# In[29]:


print(df.groupby('Startup Name')['Gross Margin'].max().nlargest(15))
fig = px.bar(df,x=df.groupby('Startup Name')['Gross Margin'].max().nlargest(15).index,y=df.groupby('Startup Name')['Gross Margin'].max().nlargest(15),
             color=df.groupby('Startup Name')['Gross Margin'].max().nlargest(15).index,title="Top 15 Startups which have highest Gross Margin [in Lakhs]")
fig.update_xaxes(title_text='Startups')
fig.update_yaxes(title_text='Gross Margin (lakhs/Yearly)')
fig.show()


# In[30]:


print(df.groupby('Startup Name')['Monthly Sales'].max().nlargest(15))
fig = px.bar(df,x=df.groupby('Startup Name')['Monthly Sales'].max().nlargest(15).index,y=df.groupby('Startup Name')['Monthly Sales'].max().nlargest(15),
             color=df.groupby('Startup Name')['Monthly Sales'].max().nlargest(15).index,title="Top 15 Startups which have highest Gross Margin [in Millons]")
fig.update_xaxes(title_text='Startups')
fig.update_yaxes(title_text='Monthly Sales')
fig.show()


# In[31]:


print(df.isnull().sum())
sns.heatmap(df.isnull())


# In[33]:


print(df.groupby('Startup Name')['Original Ask Amount'].max().nlargest(15))


# In[34]:


fig = px.bar(df,x=df.groupby('Startup Name')['Original Ask Amount'].max().nlargest(15).index,y=df.groupby('Startup Name')['Original Ask Amount'].max().nlargest(15),
             color=df.groupby('Startup Name')['Original Ask Amount'].max().nlargest(15).index,title="Top 15 Startups which ask for highest amount [in Lakhs]")
fig.update_xaxes(title_text='Startups')
fig.update_yaxes(title_text='Original Ask Amount')
fig.show()


# In[40]:


#Total Shark Deal:
total_shark=df[df["Number of sharks in deal"]==5]
figure=px.bar(total_shark, x='Startup Name', y='Total Deal Amount',title="Five Shark Deal Brands and the total investment:",text_auto=True, color='Original Ask Amount',
                    template="plotly_dark")
figure.show()


# In[50]:


print(df.groupby('Startup Name')['Original Ask Amount'].min().nsmallest(15))


# In[51]:


fig = px.bar(df,x=df.groupby('Startup Name')['Original Ask Amount'].min().nsmallest(15).index,y=df.groupby('Startup Name')['Original Ask Amount'].max().nlargest(15),
             color=df.groupby('Startup Name')['Original Ask Amount'].min().nsmallest(15).index,title="Top 15 Startups which ask for lowest amount [in Lakhs]")
fig.update_xaxes(title_text='Startups')
fig.update_yaxes(title_text='Original Ask Amount')
fig.show()


# In[77]:


#Ask Equity and Deal Equity of Highest Pitch Ask AMount Brand
figure=px.bar(df, x='Startup Name', y='Original Ask Equity',title="Ask Equity & Deal Equity of Highest Pitch Ask Amount Brand:",text_auto=True, color='Total Deal Equity',)
figure.show()


# In[91]:


#the number of investments done by individual shark
Ashneer_amount=df.loc[(df["Ashneer Investment Amount"].isnull()==False)&(df["Ashneer Investment Amount"]!=0)]
Namita_amount=df.loc[(df["Namita Investment Amount"].isnull()==False)&(df["Namita Investment Amount"]!=0)]
Anupam_amount=df.loc[(df["Anupam Investment Amount"].isnull()==False)&(df["Anupam Investment Amount"]!=0)]
Vineeta_amount=df.loc[(df["Vineeta Investment Amount"].isnull()==False)&(df["Vineeta Investment Amount"]!=0)]
Aman_amount=df.loc[(df["Aman Investment Amount"].isnull()==False)&(df["Aman Investment Amount"]!=0)]
Peyush_amount=df.loc[(df["Peyush Investment Amount"].isnull()==False)&(df["Peyush Investment Amount"]!=0)]
Ghazal_amount=df.loc[(df["Ghazal Investment Amount"].isnull()==False)&(df["Ghazal Investment Amount"]!=0)]


# In[92]:


print("-"*60,"\n","Ashneer invested in",len(Ashneer_amount),"number of business in the season.")
print("Namita invested in",len(Namita_amount),"number of business in the season.")
print("Anupam invested in",len(Anupam_amount),"number of business in the season.")
print("Vineeta invested in",len(Vineeta_amount),"number of business in the season.")
print("Aman invested in",len(Aman_amount),"number of business in the season.")
print("Peyush invested in",len(Peyush_amount),"number of business in the season.")
print("Ghazal invested in",len(Ghazal_amount),"number of business in the season.","\n","-"*60)


# In[107]:


startup_count=[len(Ashneer_amount),len(Namita_amount),len(Anupam_amount),len(Vineeta_amount),len(Aman_amount),len(Peyush_amount),len(Ghazal_amount)]
name=['Ashneer','Namita','Anupam','Vineeta','Aman','Peyush','Ghazal']
dfa= {'Name':name,'Startup_count':startup_count}
plt.figure(figsize=(10,4))
plt.bar(dfa['Name'],dfa['Startup_count'])
plt.xticks(rotation=90,fontsize=15)
plt.xlabel("Name",fontsize=14)
plt.ylabel("Number of Startups",fontsize=14)
for index,d in enumerate(startup_count):
    plt.text(x=index , y =d+0.2 , s=f"{d}" , fontdict=dict(fontsize=15))
#plt.tight_layout()
plt.title("Investment in Number of startups",fontsize=15)
plt.show()


# In[108]:


#the investments  amount done by individual shark
startup_count=[Ashneer_amount["Ashneer Investment Amount"].sum(),Namita_amount["Namita Investment Amount"].sum(),Anupam_amount["Anupam Investment Amount"].sum(),Vineeta_amount["Vineeta Investment Amount"].sum(),Aman_amount["Aman Investment Amount"].sum(),Peyush_amount["Peyush Investment Amount"].sum(),Ghazal_amount["Ghazal Investment Amount"].sum()]
name=['Ashneer','Namita','Anupam','Vineeta','Aman','Peyush','Ghazal']
dfa = {'Name':name,'Startup_count':startup_count}
plt.figure(figsize=(10,4))
plt.bar(dfa['Name'],dfa['Startup_count'])
plt.xticks(rotation=90,fontsize=15)
plt.xlabel("Name",fontsize=14)
plt.ylabel("Amount in Lakh",fontsize=14)
for index,d in enumerate(startup_count):
    plt.text(x=index , y =d+1 , s=f"{round(d,2)}" ,ha = 'center', fontdict=dict(fontsize=12))
#plt.tight_layout()
plt.title("Investments done by individual shark ",fontsize=15)
plt.show()


# In[102]:


#equity %
equity=[df['Ashneer Investment Equity'].sum(), df['Namita Investment Equity'].sum(), df['Anupam Investment Equity'].sum(), df['Vineeta Investment Equity'].sum(),
    df['Aman Investment Equity'].sum(), df['Peyush Investment Equity'].sum(), df['Ghazal Investment Equity'].sum()]
name=['Ashneer','Namita','Anupam','Vineeta','Aman','Peyush','Ghazal']
df = {'Name':name,'Total equity':equity }
plt.figure(figsize=(10,4))
plt.bar(df['Name'],df['Total equity'])
plt.xticks(rotation=90,fontsize=15)
plt.xlabel("Name",fontsize=14)
plt.ylabel("Sum of equity percentage in different companies",fontsize=14)
for index,d in enumerate(equity):
    plt.text(x=index , y =d+2 , s=f"{round(d,2)}" ,ha = 'center', fontdict=dict(fontsize=12))
#plt.tight_layout()
plt.title("Total equity percentage of individual shark ",fontsize=15)
plt.show()


# In[110]:


#investment based on the  Debt and loaned Amount
debt=[df['Ashneer Debt Amount'].sum(), df['Namita Debt Amount'].sum(), df['Anupam Debt Amount'].sum(), df['Vineeta Debt Amount'].sum(),
    df['Aman Debt Amount'].sum(), df['Peyush Debt Amount'].sum(), df['Ghazal Debt Amount'].sum()]
name=['Ashneer','Namita','Anupam','Vineeta','Aman','Peyush','Ghazal']
dfa = {'Name':name,'Total debt':debt }
plt.figure(figsize=(10,4))
plt.bar(dfa['Name'],dfa['Total debt'])
plt.xticks(rotation=90,fontsize=15)
plt.xlabel("Name",fontsize=14)
plt.ylabel("Total debt amount in lakh",fontsize=14)
for index,d in enumerate(debt):
    plt.text(x=index , y =d+2 , s=f"{round(d,2)}" ,ha = 'center', fontdict=dict(fontsize=12))
#plt.tight_layout()
plt.title("Debt amount given by individual shark",fontsize=15)
plt.show()


# In[111]:


#
DealDone=df.loc[df["Accepted Offer"]==1]

fig = plt.figure(figsize=(15,8))    

plt.subplot()
plt.scatter("Total Deal Equity","Total Deal Amount",data=DealDone,s="Valuation Offered",c="Valuation Offered",cmap="flare",edgecolor="k",linewidths=.8)

plt.title("Total Investment and equity",fontsize=20)
plt.xlabel("Equity in %",fontsize=15,color="k")
plt.ylabel("Investmet in Lacks",fontsize=15,color="k")
plt.colorbar(label="valuation in Lacks")


# ## Ashneer's Investment Analysis

# In[112]:


## Ashneer's Investment Analysis
Ashneer = Ashneer_amount.sort_values('Ashneer Investment Amount')
plt.rcParams.update({'font.size': 16})

plt.figure(figsize=(35,20))
plt.subplot(2,2,1)
plt.barh(Ashneer["Startup Name"],Ashneer["Ashneer Investment Amount"])
for i, v in enumerate(Ashneer["Ashneer Investment Amount"]):
    plt.text(v, i , str(round(v))+"L", color='black', fontweight='bold')
plt.title("Ashneer Investment Amount")
plt.subplot(2,2,2)
plt.barh(Ashneer["Startup Name"],Ashneer["Ashneer Investment Equity"])
for i, v in enumerate(Ashneer["Ashneer Investment Equity"]):
    plt.text(v, i , str(round(v))+"%", color='black', fontweight='bold')
plt.title("Ashneer Investment Equity")
plt.subplot(2,2,3)
Ashneer["Industry"].value_counts().plot(kind='pie',autopct='%.2f%%',shadow=True)
plt.title("Industry")
plt.subplot(2,2,4)
plt.barh(Ashneer["Startup Name"],Ashneer["Number of sharks in deal"])
for i, v in enumerate(Ashneer["Number of sharks in deal"]):
    plt.text(v, i , str(round(v)), color='black', fontweight='bold')
plt.title("Number of sharks in deal")
plt.suptitle("Ashneer's Investment Analysis",fontsize=40)


# ## Namita Investment Analysis

# In[113]:


Namita = Namita_amount.sort_values('Namita Investment Amount')
plt.rcParams.update({'font.size': 16})

plt.figure(figsize=(35,20))
plt.subplot(2,2,1)
plt.barh(Namita["Startup Name"],Namita["Namita Investment Amount"])
for i, v in enumerate(Namita["Namita Investment Amount"]):
    plt.text(v, i , str(round(v))+"L", color='black', fontweight='bold')
plt.title("Namita Investment Amount")
plt.subplot(2,2,2)
plt.barh(Namita["Startup Name"],Namita["Namita Investment Equity"])
for i, v in enumerate(Namita["Namita Investment Equity"]):
    plt.text(v, i , str(round(v))+"%", color='black', fontweight='bold')
plt.title("Namita Investment Equity")
plt.subplot(2,2,3)
Namita["Industry"].value_counts().plot(kind='pie',autopct='%.2f%%',shadow=True)
plt.title("Industry")
plt.subplot(2,2,4)
plt.barh(Namita["Startup Name"],Namita["Number of sharks in deal"])
for i, v in enumerate(Namita["Number of sharks in deal"]):
    plt.text(v, i , str(round(v)), color='black', fontweight='bold')
plt.title("Number of sharks in deal")
plt.suptitle("Namita's Investment Analysis",fontsize=40)


# ## Anupam Investment Analysis

# In[114]:


Anupam = Anupam_amount.sort_values('Anupam Investment Amount')
plt.rcParams.update({'font.size': 15})

plt.figure(figsize=(38,25))
plt.subplot(2,2,1)
plt.barh(Anupam["Startup Name"],Anupam["Anupam Investment Amount"])
for i, v in enumerate(Anupam["Anupam Investment Amount"]):
    plt.text(v, i , str(round(v))+"L", color='black', fontweight='bold')
plt.title("Anupam Investment Amount")
plt.subplot(2,2,2)
plt.barh(Anupam["Startup Name"],Anupam["Anupam Investment Equity"])
for i, v in enumerate(Anupam["Anupam Investment Equity"]):
    plt.text(v, i , str(round(v))+"%", color='black', fontweight='bold')
plt.title("Anupam Investment Equity")
plt.subplot(2,2,3)
Anupam["Industry"].value_counts().plot(kind='pie',autopct='%.2f%%',shadow=True)
plt.title("Industry")
plt.subplot(2,2,4)
plt.barh(Anupam["Startup Name"],Anupam["Number of sharks in deal"])
for i, v in enumerate(Anupam["Number of sharks in deal"]):
    plt.text(v, i , str(round(v)), color='black', fontweight='bold')
plt.title("Number of sharks in deal")
plt.suptitle("Anupam's Investment Analysis",fontsize=40)


# ## 'Vineeta Investment Amount'

# In[115]:


Vineeta = Vineeta_amount.sort_values('Vineeta Investment Amount')
plt.rcParams.update({'font.size': 16})

plt.figure(figsize=(35,20))
plt.subplot(2,2,1)
plt.barh(Vineeta["Startup Name"],Vineeta["Vineeta Investment Amount"])
for i, v in enumerate(Vineeta["Vineeta Investment Amount"]):
    plt.text(v, i , str(round(v))+"L", color='black', fontweight='bold')
plt.title("Vineeta Investment Amount")
plt.subplot(2,2,2)
plt.barh(Vineeta["Startup Name"],Vineeta["Vineeta Investment Equity"])
for i, v in enumerate(Vineeta["Vineeta Investment Equity"]):
    plt.text(v, i , str(round(v))+"%", color='black', fontweight='bold')
plt.title("Vineeta Investment Equity")
plt.subplot(2,2,3)
Vineeta["Industry"].value_counts().plot(kind='pie',autopct='%.2f%%',shadow=True)
plt.title("Industry")
plt.subplot(2,2,4)
plt.barh(Vineeta["Startup Name"],Vineeta["Number of sharks in deal"])
for i, v in enumerate(Vineeta["Number of sharks in deal"]):
    plt.text(v, i , str(round(v)), color='black', fontweight='bold')
plt.title("Number of sharks in deal")
plt.suptitle("Vineeta's Investment Analysis",fontsize=40)


# ## Aman Investment Amount

# In[116]:


Aman = Aman_amount.sort_values('Aman Investment Amount')
plt.rcParams.update({'font.size': 16})

plt.figure(figsize=(35,20))
plt.subplot(2,2,1)
plt.barh(Aman["Startup Name"],Aman["Aman Investment Amount"])
for i, v in enumerate(Aman["Aman Investment Amount"]):
    plt.text(v, i , str(round(v))+"L", color='black', fontweight='bold')
plt.title("Aman Investment Amount")
plt.subplot(2,2,2)
plt.barh(Aman["Startup Name"],Aman["Aman Investment Equity"])
for i, v in enumerate(Aman["Aman Investment Equity"]):
    plt.text(v, i , str(round(v))+"%", color='black', fontweight='bold')
plt.title("Aman Investment Equity")
plt.subplot(2,2,3)
Aman["Industry"].value_counts().plot(kind='pie',autopct='%.2f%%',shadow=True)
plt.title("Industry")
plt.subplot(2,2,4)
plt.barh(Aman["Startup Name"],Aman["Number of sharks in deal"])
for i, v in enumerate(Aman["Number of sharks in deal"]):
    plt.text(v, i , str(round(v)), color='black', fontweight='bold')
plt.title("Number of sharks in deal")
plt.suptitle("Aman's Investment Analysis",fontsize=40)


# ## Peyush Investment Amount

# In[118]:


##Peyush's Investment Analysis
Peyush = Peyush_amount.sort_values('Peyush Investment Amount')
plt.rcParams.update({'font.size': 16})

plt.figure(figsize=(35,20))
plt.subplot(2,2,1)
plt.barh(Peyush["Startup Name"],Peyush["Peyush Investment Amount"])
for i, v in enumerate(Peyush["Peyush Investment Amount"]):
    plt.text(v, i , str(round(v))+"L", color='black', fontweight='bold')
plt.title("Peyush Investment Amount")
plt.subplot(2,2,2)
plt.barh(Peyush["Startup Name"],Peyush["Peyush Investment Equity"])
for i, v in enumerate(Peyush["Peyush Investment Equity"]):
    plt.text(v, i , str(round(v))+"%", color='black', fontweight='bold')
plt.title("Peyush Investment Equity")
plt.subplot(2,2,3)
Peyush["Industry"].value_counts().plot(kind='pie',autopct='%.2f%%',shadow=True)
plt.title("Industry")
plt.subplot(2,2,4)
plt.barh(Peyush["Startup Name"],Peyush["Number of sharks in deal"])
for i, v in enumerate(Peyush["Number of sharks in deal"]):
    plt.text(v, i , str(round(v)), color='black', fontweight='bold')
plt.title("Number of sharks in deal")
plt.suptitle("Peyush's Investment Analysis",fontsize=40)


# ## Ghazal Investment Amount

# In[119]:


Ghazal = Ghazal_amount.sort_values('Ghazal Investment Amount')
plt.rcParams.update({'font.size': 16})

plt.figure(figsize=(35,20))
plt.subplot(2,2,1)
plt.barh(Ghazal["Startup Name"],Ghazal["Ghazal Investment Amount"])
for i, v in enumerate(Ghazal["Ghazal Investment Amount"]):
    plt.text(v, i , str(round(v))+"L", color='black', fontweight='bold')
plt.title("Ghazal Investment Amount")
plt.subplot(2,2,2)
plt.barh(Ghazal["Startup Name"],Ghazal["Ghazal Investment Equity"])
for i, v in enumerate(Ghazal["Ghazal Investment Equity"]):
    plt.text(v, i , str(round(v))+"%", color='black', fontweight='bold')
plt.title("Ghazal Investment Equity")
plt.subplot(2,2,3)
Ghazal["Industry"].value_counts().plot(kind='pie',autopct='%.2f%%',shadow=True)
plt.title("Industry")
plt.subplot(2,2,4)
plt.barh(Ghazal["Startup Name"],Ghazal["Number of sharks in deal"])
for i, v in enumerate(Ghazal["Number of sharks in deal"]):
    plt.text(v, i , str(round(v)), color='black', fontweight='bold')
plt.title("Number of sharks in deal")
plt.suptitle("Ghazal's Investment Analysis",fontsize=40)


# In[ ]:




