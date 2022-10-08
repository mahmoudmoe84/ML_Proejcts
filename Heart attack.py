#!/usr/bin/env python
# coding: utf-8

# In[561]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[562]:


cleve = pd .read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")
                     


# In[563]:


cleve.columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num_the predicted attribute'] 


# In[564]:


cleve


# In[565]:


hungar = pd .read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data") 
hungar.columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num_the predicted attribute'] 
hungar


# In[566]:


switze = pd .read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data") 
switze.columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num_the predicted attribute'] 
switze


# In[567]:


va = pd .read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data")
va.columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num_the predicted attribute'] 
va


# In[448]:


data= pd.concat([cleve,va,switze,hungar])
data


# In[449]:


data.info()


# In[450]:


data =data.replace("?",np.nan)


# In[451]:


data.isna().sum()


# In[452]:


data.info()


# In[453]:


sns.heatmap(data.isnull(),cmap="viridis")


# In[454]:


data.isna().sum()/data.shape[0]


# In[455]:


data['trestbps'] = data['trestbps'].astype(float)
data['chol'] = data['chol'].astype(float)
data['fbs'] = data['fbs'].astype(float)
data['restecg']= data['restecg'].astype(float)
data['thalach']= data['thalach'].astype(float)
data['exang']= data['exang'].astype(float)
data['trestbps']= data['trestbps'].astype(float)
data['oldpeak']= data['oldpeak'].astype(float)                   
data['slope']= data['slope'].astype(float)
data['ca']= data['ca'].astype(float)                   
data['thal']= data['thal'].astype(float)    


# In[456]:


plt.figure(figsize=(15,10))
sns.heatmap(data.corr(),annot=True)


# In[457]:



data['num_the predicted attribute']


# In[479]:


result =[]
for value in data["num_the predicted attribute"]:
    if value == 0:
        result.append(0)
    elif value == "NaN":
        result.append('NaN')
    else:
        result.append(1)  
    


# In[480]:


df = pd.DataFrame({"new_num_the predicted attribute":result})
df


# In[481]:


data['new_num_the predicted attribute']= df


# In[482]:


data = data.drop(columns="num_the predicted attribute") 
data


# In[483]:


data['new_num_the predicted attribute'].value_counts()


# # handling data by using StandardScaler

# In[484]:


data_new = data.dropna()


# In[485]:


from sklearn.preprocessing import StandardScaler


# In[486]:


sc = StandardScaler()


# In[487]:


sc.fit_transform( data_new)


# # splitting and training the data 

# In[488]:


X['sex'] = X['sex'].astype(int)
X['cp'] = X['cp'].astype(int)
X['exang']= X['exang'].astype(int)               
X['slope']= X['slope'].astype(int)
X['ca']= X['ca'].astype(int)                   


# In[489]:


X =data_new[["sex","cp","exang","ca","slope"]]


# In[490]:


X.info()


# In[491]:


X.values


# In[492]:


y = data_new["thal"]


# In[493]:


y = y.astype(int)
y.values


# In[494]:


from sklearn.model_selection import train_test_split 


# In[495]:


X_test,X_train,y_test,y_train =train_test_split(X,y,test_size=0.6,random_state=42)


# # handling missing data using SVM (thal)

# In[496]:


from sklearn.svm import SVC


# In[497]:


svc = SVC()


# In[498]:


model = svc.fit(X_train,y_train)


# In[499]:


y_predict= model.predict(X_test)


# In[500]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[501]:


confusion_matrix(y_test,y_predict)


# In[502]:


accuracy_score(y_test,y_predict)


# # handling missing data using Logistic Regression (thal)

# In[503]:


from sklearn.linear_model import LogisticRegression 


# In[504]:


Lr= LogisticRegression ()


# In[505]:


model_1 =Lr.fit(X_train,y_train)


# In[506]:


y_predict= model_1.predict(X_test)


# In[507]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[508]:


accuracy_score(y_test,y_predict)


# # handling missing data using naive_bayes (thal)

# In[509]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[510]:


from sklearn.naive_bayes import MultinomialNB


# In[511]:


mul = MultinomialNB()


# In[512]:


model_3= mul.fit(X_train,y_train)


# In[513]:


y_predict= model_3.predict(X_test)


# In[514]:



accuracy_score(y_test,y_predict)


# # handling missing data using Kmeans  (thal)

# In[515]:


from sklearn.neighbors import KNeighborsClassifier


# In[516]:


clf= KNeighborsClassifier(n_neighbors=27,metric='minkowski',p=7)


# In[517]:


clf.fit(X_train,y_train)


# In[518]:


y_predict =clf.predict(X_test)
y_predict


# In[519]:


accuracy_score(y_test,y_predict)


# # handling missing data using Scikit learn Knnimputer (thal)

# In[520]:


data_KNN_Imputer = data.copy(deep=True)
data_KNN_Imputer


# In[521]:


from sklearn.impute import KNNImputer


# In[522]:


knnimputer = KNNImputer(n_neighbors=3)


# In[523]:


data_KNN_Imputer.iloc[:,:] = knnimputer.fit_transform(data_KNN_Imputer)


# In[524]:


data_KNN_Imputer


# In[525]:


data_KNN_Imputer.isna().sum()


# In[526]:


data_KNN_Imputer['thal']


# # handling missing data using KNNFancyimputer (thal)

# In[527]:


pip install fancyimpute


# In[528]:


knn_fancy_imputer =data.copy(deep=True)
knn_fancy_imputer


# In[529]:


knn_fancy_imputer.isna().sum()


# In[530]:


from fancyimpute import KNN


# In[531]:


knn_1 = KNN()


# In[532]:


knn_fancy_imputer.iloc[:,:]= knn_1.fit_transform(knn_fancy_imputer)


# In[533]:


knn_fancy_imputer


# # handling missing data using DecisionTreeClassifier (thal)

# In[534]:


from sklearn.tree import DecisionTreeClassifier


# In[535]:


tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)


# In[536]:


tree.fit(X_train,y_train)


# In[537]:


clf =tree.fit(X_train,y_train)
clf


# In[538]:


predict = clf.predict(X_test)


# In[539]:


accuracy_score(y_test,y_predict)


# # handling missing data  using interpolation (numerical data)

# In[540]:


new_data_1 = data
data


# In[541]:


s= data[["trestbps","chol","thalach","age","oldpeak"]]
s


# In[542]:


s.isna().sum()


# In[543]:


s_1=s.interpolate()
s_1


# In[544]:


s_1.isna().sum()


# In[545]:


d= s.compare(s_1,keep_shape=True,keep_equal=True)
d


# # handling missing data  using interpolation (catagerocial data)

# In[546]:


cata = data[["sex","cp","fbs","restecg","exang","slope","ca","thal","new_num_the predicted attribute"]]
cata


# In[547]:


cata.isna().sum()


# In[548]:


cata["ca"].value_counts()


# In[431]:


cata_1 = cata.interpolate()
cata_1  


# In[432]:


cata_1["ca"].value_counts()


# # missing value using EM 

# In[549]:


pip install impyute


# In[550]:


import impyute as impy
X_theme = data[['thal']]
X_rand = data[['thal']]
df_theme_missing1 = data.copy()
df_rand_missing1 = data.copy()
data[['thal']] = impy.em(df_theme_missing1[['thal']].values, loops=500)
data[['thal']] = impy.em(df_rand_missing1[['thal']].values, loops=500)


# In[551]:


data[['thal']].isna().sum()


# In[552]:


data.isna().sum()


# In[553]:


data[['thal']]


# # handling missing data using charts 

# In[558]:


sns.countplot(data=data, x="thal", hue="new_num_the predicted attribute")


# In[ ]:





# In[ ]:




