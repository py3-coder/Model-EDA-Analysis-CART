#!/usr/bin/env python
# coding: utf-8

# # Campus Recruitment DRIVE

# In[55]:


## Author : Saurabh


# In[1]:


pwd


# In[2]:


path ='E:\\DataScience\\MachineLearning\\Campus Recruitment'


# In[3]:


import os
os.listdir()


# In[4]:


# Import all the tools we need

# Regular EDA (exploratory data analysis) and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# we want our plots to appear inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score ,accuracy_score
from sklearn.metrics import plot_roc_curve

#Ignore Warnings
import warnings
warnings.filterwarnings('ignore')


# In[5]:


#Importing data 
df = pd.read_csv(path+"/Placement_Data_Full_Class.csv")
# We do not need salary column so we can delete it
## visulization task
placement_copy=df.copy()
df= df.drop("salary" , axis =1)
df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


#checking null values
df.isna().sum()


# In[9]:


#Which Department students have good chance of getting placed
df.groupby('degree_t')['status'].value_counts()


# In[13]:


#Based on Gender who are getting good placement?
df.groupby('gender')['status'].value_counts()


# In[14]:


df['gender'].value_counts()


# In[17]:


#Does WorkExperience matters in placement
df.groupby('workex')['status'].value_counts()


# In[20]:


#Does Employability test percentage conducted by college matters?
groups = df.groupby(['status', pd.cut(df.etest_p, [40,60,80, 100])])

#display bin count grouped by team
groups.size().unstack()


# In[21]:


#Does High school percentage matters?
groups = df.groupby(['status', pd.cut(df.hsc_p, [10, 50, 70, 100])])

#display bin count grouped by team
groups.size().unstack()


# ## Basic data visualization

# In[22]:


placement_copy.head()


# In[23]:


placement_copy['salary'].fillna(value=0, inplace=True)
print('Salary column with null values:',placement_copy['salary'].isnull().sum(), sep = '\n')


# In[24]:


placement_copy.drop(['sl_no','ssc_b','hsc_b'], axis = 1,inplace=True) 
placement_copy.head()


# In[25]:


plt.figure(figsize = (15, 10))
plt.style.use('seaborn-white')
ax=plt.subplot(221)
plt.boxplot(placement_copy['ssc_p'])
ax.set_title('Secondary school percentage')
ax=plt.subplot(222)
plt.boxplot(placement_copy['hsc_p'])
ax.set_title('Higher Secondary school percentage')
ax=plt.subplot(223)
plt.boxplot(placement_copy['degree_p'])
ax.set_title('UG Degree percentage')
ax=plt.subplot(224)
plt.boxplot(placement_copy['etest_p'])
ax.set_title('Employability percentage')


# In[26]:


Q1 = placement_copy['hsc_p'].quantile(0.25)
Q3 = placement_copy['hsc_p'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

filter = (placement_copy['hsc_p'] >= Q1 - 1.5 * IQR) & (placement_copy['hsc_p'] <= Q3 + 1.5 *IQR)
placement_filtered=placement_copy.loc[filter]


# In[27]:


plt.figure(figsize = (15, 5))
plt.style.use('seaborn-white')
ax=plt.subplot(121)
plt.boxplot(placement_copy['hsc_p'])
ax.set_title('Before removing outliers(hsc_p)')
ax=plt.subplot(122)
plt.boxplot(placement_filtered['hsc_p'])
ax.set_title('After removing outliers(hsc_p)')


# In[28]:


## COUNT PLOTS


# In[29]:


plt.figure(figsize = (15, 7))
plt.style.use('seaborn-white')

#Specialisation
plt.subplot(234)
ax=sns.countplot(x="specialisation", data=placement_filtered, facecolor=(0, 0, 0, 0),
                 linewidth=5,edgecolor=sns.color_palette("magma", 3))
fig = plt.gcf()
fig.set_size_inches(10,10)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)

#Work experience
plt.subplot(235)
ax=sns.countplot(x="workex", data=placement_filtered, facecolor=(0, 0, 0, 0),
                 linewidth=5)
fig = plt.gcf()
fig.set_size_inches(10,10)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)

#Degree type
plt.subplot(233)
ax=sns.countplot(x="degree_t", data=placement_filtered, facecolor=(0, 0, 0, 0),
                 linewidth=5)
fig = plt.gcf()
fig.set_size_inches(10,10)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12,rotation=20)

#Gender
plt.subplot(231)
ax=sns.countplot(x="gender", data=placement_filtered, facecolor=(0, 0, 0, 0),
                 linewidth=5)
fig = plt.gcf()
fig.set_size_inches(10,10)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)

#Higher secondary specialisation
plt.subplot(232)
ax=sns.countplot(x="hsc_s", data=placement_filtered, facecolor=(0, 0, 0, 0),
                 linewidth=5)
fig = plt.gcf()
fig.set_size_inches(10,10)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)

#Status of recruitment
plt.subplot(236)
ax=sns.countplot(x="status", data=placement_filtered, facecolor=(0, 0, 0, 0),
                 linewidth=5)
fig = plt.gcf()
fig.set_size_inches(10,10)
ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)


# In[30]:


#Distribution Salary- Placed Students
sns.set(rc={'figure.figsize':(12,8)})
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

placement_placed = placement_filtered[placement_filtered.salary != 0]
sns.boxplot(placement_placed["salary"], ax=ax_box)
sns.distplot(placement_placed["salary"], ax=ax_hist)
 
# Remove x axis name for the boxplot
ax_box.set(xlabel='')


# In[31]:


#Distribution of all percentages
plt.figure(figsize = (15, 7))
plt.style.use('seaborn-white')
plt.subplot(231)
sns.distplot(placement_filtered['ssc_p'])
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(232)
sns.distplot(placement_filtered['hsc_p'])
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(233)
sns.distplot(placement_filtered['degree_p'])
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(234)
sns.distplot(placement_filtered['etest_p'])
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(235)
sns.distplot(placement_filtered['mba_p'])
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(236)
sns.distplot(placement_placed['salary'])
fig = plt.gcf()
fig.set_size_inches(10,10)


# In[32]:


#Work experience Vs Placement Status

plt.style.use('seaborn-white')
f,ax=plt.subplots(1,2,figsize=(18,8))
placement_filtered['workex'].value_counts().plot.pie(explode=[0,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Work experience')
sns.countplot(x = 'workex',hue = "status",data = placement_filtered)
ax[1].set_title('Influence of experience on placement')
plt.show()


# In[33]:


#MBA marks vs Placement Status- Does your academic score influence?
g = sns.boxplot(y = "status",x = 'mba_p',data = placement_filtered, whis=np.inf)
g = sns.swarmplot(y = "status",x = 'mba_p',data = placement_filtered, size = 7,color = 'black')
sns.despine()
g.figure.set_size_inches(12,8)
plt.show()


# In[34]:


#Does MBA percentage and Employability score correlate?

gapminder=px.data.gapminder()
px.scatter(placement_filtered,x="mba_p",y="etest_p",color="status",facet_col="workex")


# In[35]:


#Is there any gender bias while offering remuneration?
px.violin(placement_placed,y="salary",x="specialisation",color="gender",box=True,points="all")


# In[36]:


#Coorelation between academic percentages
sns.heatmap(placement_placed.corr(),annot=True,fmt='.1g',cmap='Greys')


# In[37]:


#Distribution of our data
sns.pairplot(placement_filtered,vars=['ssc_p','hsc_p','degree_p','mba_p','etest_p'],hue="status")


# In[40]:


#converting 'placed'= 1 , "Not placed" = 0
df['status'] = df['status'].replace(['Placed'],1)
df['status'] = df['status'].replace(['Not Placed'],0)


df['gender'] = df['gender'].replace(['M'],1)
df['gender'] = df['gender'].replace(['F'],0)


df['hsc_b'] = df['hsc_b'].replace(['Others'],1)
df['hsc_b'] = df['hsc_b'].replace(['Central'],2)

df['ssc_b'] = df['ssc_b'].replace(['Others'],1)
df['ssc_b'] = df['ssc_b'].replace(['Central'],2)


df['workex'] = df['workex'].replace(['Yes'],1)
df['workex'] = df['workex'].replace(['No'],0)

df['hsc_s'] = df['hsc_s'].replace(['Commerce'],1)
df['hsc_s'] = df['hsc_s'].replace(['Science'],2)
df['hsc_s'] = df['hsc_s'].replace(['Arts'],3)

df['specialisation'] = df['specialisation'].replace(['Mkt&HR'],1)
df['specialisation'] = df['specialisation'].replace(['Mkt&Fin'],2)


df['degree_t'] = df['degree_t'].replace(['Comm&Mgmt'],1)
df['degree_t'] = df['degree_t'].replace(['Sci&Tech'],2)
df['degree_t'] = df['degree_t'].replace(['Others'],3)


# In[41]:


#converting object into int
df1= df
df1


# In[42]:


# Splitting data into X and y

X = df1.drop(["sl_no","status"], axis=1)
y = df1["status"]


# In[43]:


# Split data into train and test sets

np.random.seed(50)
# Split into train & test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)


# In[44]:


# Put models in a dictionary
models = {"Logistic Regression": LogisticRegression(solver='liblinear'),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier(class_weight = {1:0.1, 0:0.9})}

# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of differetn Scikit-Learn machine learning models
    X_train : training data (no labels)
    X_test : testing data (no labels)
    y_train : training labels
    y_test : test labels
    """
    # Set random seed
    np.random.seed(50)
    # Make a dictionary to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores


# In[45]:


model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)

model_scores


# In[46]:


model_compare = pd.DataFrame(model_scores, index=["accuracy"])
show_values(model_compare.T.plot.bar());


# ## Stratified K-fold Cross Validation

# In[47]:


from sklearn.model_selection import StratifiedKFold
skfold=StratifiedKFold(n_splits=5)
model1=LogisticRegression()
scores1=cross_val_score(model1,X,y,cv=skfold)
print(np.mean(scores1))


# In[48]:


from sklearn.model_selection import StratifiedKFold
skfold=StratifiedKFold(n_splits=5)
model2=RandomForestClassifier()
scores2=cross_val_score(model2,X,y,cv=skfold)
print(np.mean(scores2))


# ## LOGISTIC REGRESSION

# In[49]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)


# In[50]:


classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[51]:


y_pred = classifier.predict(X_test)


# In[52]:


print(classification_report(y_test, y_pred))


# In[53]:


print("Accuracy Score {}".format(accuracy_score(y_test,y_pred)))


# In[54]:


# Plot ROC curve and calculate and calculate AUC metric
plot_roc_curve(classifier, X_test, y_test)


# In[ ]:


df


# ## CONCLUSION : LOGISTIC REGRESSION ~ Accuracy - 92%

# In[ ]:




