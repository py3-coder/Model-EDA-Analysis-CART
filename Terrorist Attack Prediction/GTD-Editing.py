#Importing Libraries
import numpy as np
import pandas as pd
#importing the raw Global Terrorism Database and storing it in datsaset
dataset = pd.read_csv("/Users/kushankurghosh/Documents/ML(IT)- Files/Project/Global Terrorism Database.csv")
dataset.isnull().sum()
#Dropping Columns one by one from dataset where the csv file is stored 
dataset.drop(['approxdate','resolution','location','alternative','alternative_txt','propextent','propextent_txt','propvalue','propcomment','nhostkid','nhostkidus','nhours','ndays','divert','kidhijcountry','ransom','ransomamt','ransomamtus','ransompaid','ransompaidus','ransomnote','hostkidoutcome','hostkidoutcome_txt','nreleased','addnotes','scite2','scite3','related'], axis=1, inplace=True)
dataset.drop(['claim2','claimmode2','claimmode2_txt','claim3','claimmode3','claimmode3_txt','compclaim','weaptype2','weaptype2_txt','weapsubtype2','weapsubtype2_txt'],axis=1,inplace=True)
dataset.drop(['gsubname3','motive','guncertain2','guncertain3','claimmode','claimmode_txt'],axis=1,inplace=True)
dataset.drop(['natlty3_txt','gsubname','gname2','gsubname2','gname3'],axis=1,inplace=True)
dataset.drop(['targsubtype3','targsubtype3_txt','corp3','target3','natlty3'],axis=1,inplace=True)
dataset.drop(['target2','natlty2','natlty2_txt','targtype3','targtype3_txt'],axis=1,inplace=True)
dataset.drop(['targtype2','targtype2_txt','targsubtype2','targsubtype2_txt','corp2'],axis=1,inplace=True)
dataset.drop(['nwoundus','nkillus'],axis=1,inplace=True)
dataset.drop(['nwoundte','nkillter'],axis=1,inplace=True)
dataset.drop(['weapdetail','crit1','crit2','crit3'],axis=1,inplace=True)
dataset.drop(['summary','claimed','INT_LOG','INT_IDEO','INT_MISC','INT_ANY','scite1','multiple'],axis=1,inplace=True)
dataset.drop(['specificity','vicinity','doubtterr','corp1','nperpcap','dbsource'],axis=1,inplace=True)
dataset.drop(['iyear','imonth','iday','eventid','extended'],axis=1,inplace=True)
dataset.drop(['latitude','longitude','ishostkid','country_txt','region_txt','city','attacktype1_txt','targtype1_txt','targsubtype1_txt','natlty1_txt','weapsubtype1_txt','weaptype1_txt','target1'],axis=1,inplace=True)
dataset.drop(['individual','guncertain1','nperps'],axis=1,inplace=True)
dataset.drop(['country','region','provstate','success','suicide','natlty1','gname','property'],axis=1,inplace=True)
#dropping the rows with Nan values for the remaining columns 
dataset.dropna(how='any',inplace=True)
#Converting the obtained dataframe into a csv file by calling it GTD_Purified.csv
dataset.to_csv("/Users/kushankurghosh/Documents/ML IT- Files/Project/GTD_Purified.csv")
