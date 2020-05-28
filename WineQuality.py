#!/usr/bin/env python
# coding: utf-8

# In[1]:


#I - importing libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[2]:


#II - importing dataset
db = pd.read_csv('winequality.csv')


# In[3]:


db


# In[4]:


db.head(5)


# In[5]:


db.tail(5)


# In[6]:


db.info()


# In[7]:


#Plotting for visualisation purposes - fixed_acidityVsQuality
fig = plt.figure(figsize = (15, 10))
sns.barplot(x = 'quality', y = 'fixed acidity', data = db)

#We see that fixed acidity does not give any specification on quality.


# In[8]:


#volatile_acidityVsQuality
fig = plt.figure(figsize = (15,10))
sns.barplot(x = 'quality', y = 'volatile acidity', data = db)

#Here we see that as wine quality decreases, volatile acidity increases.


# In[9]:


#Citric_acidVsQuality
fig = plt.figure(figsize = (15,10))
sns.barplot(x = 'quality', y = 'citric acid', data = db)

#Here we see that as wine quality increases citric acidity also increases.


# In[10]:


#QualityVsResidualSugar
fig = plt.figure(figsize = (15,10))
sns.barplot(x = 'quality', y = 'residual sugar', data = db)

#here we see that quality of wine and residual sugar don't affect each other marginally.


# In[11]:


#QualityVsChlorides
fig = plt.figure(figsize = (15,10))
sns.barplot(x = 'quality', y = 'chlorides', data = db)

#Here we see as quality increases, chloride composition decreases.


# In[12]:


#QualityVsSulphates
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = db)

#Here we see that as quality increases so does sulphate level.


# In[13]:


#Preprocessing the data for the algorithms
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
db['quality'] = pd.cut(db['quality'], bins = bins, labels = group_names)


# In[14]:


from sklearn.preprocessing import StandardScaler, LabelEncoder
label_quality = LabelEncoder()


# In[15]:


db['quality'] = label_quality.fit_transform(db['quality'])


# In[16]:


sns.countplot(db['quality'])


# In[17]:


X = db.drop('quality', axis = 1)
y = db['quality']


# In[18]:


#III - training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# In[19]:


sc = StandardScaler()


# In[20]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[21]:


#IV - training the dataset on the algorithm - kernel SVM
from sklearn.svm import SVC
sv = SVC(kernel = 'rbf', random_state = 0)
sv.fit(X_train, y_train)
pred_sv = sv.predict(X_test)


# In[22]:


from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, pred_sv)
print(classification_report(y_test, pred_sv))

#Here we see that when we use SVM model we get an f1 score of 86%


# In[23]:


#training with NaiveBayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)


# In[24]:


cm = confusion_matrix(y_test, pred_nb)
print(classification_report(y_test, pred_nb))

#The naiveBayes Model gives a f1 score of 86%


# In[25]:


#training with RandomForest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)


# In[26]:


cm = confusion_matrix(y_test, pred_rf)
print(classification_report(y_test, pred_rf))

#Random forest gives us a f1 rate of 89%. Try changing the estimators and check for better results.


# In[27]:


#training with kNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)


# In[28]:


cm = confusion_matrix(y_test, pred_knn)
print(classification_report(y_test, pred_knn))

#Via kNN we get an accuracy of 89%.


# In[ ]:




