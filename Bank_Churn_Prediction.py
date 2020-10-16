#!/usr/bin/env python
# coding: utf-8
Objective:
Given a Bank customer, build a neural network based classifier that can determine whether they will leave
or not in the next 6 months.

Context:
Businesses like banks which provide service have to worry about problem of 'Churn' i.e. customers
leaving and joining another service provider. It is important to understand which aspects of the service
influence a customer's decision in this regard. Management can concentrate efforts on improvement of
service, keeping in mind these priorities.

Data Description:
The case study is from an open-source dataset from Kaggle.
The dataset contains 10,000 sample points with 14 d
# In[292]:


import tensorflow as tf
print(tf.__version__)


# In[293]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers


# In[294]:


# Loading data
dataset  = pd.read_csv("bank.csv")


# In[295]:


dataset.head()


# In[296]:


#Checking unique values to decide categorical attributes vs others
dataset.nunique()


# In[297]:


# Checking null values and datatypes
dataset.info()


# In[298]:


# Drop RowNumber and CustomerId because it won't be useful in predictive task
dataset.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)


# In[299]:


# Convert object dtype into category
cat_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'NumOfProducts', 'Tenure']
for colname in cat_features:
    dataset[colname] = dataset[colname].astype('category')


# In[300]:


#
dataset.columns
num_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']


# In[301]:


#Normalizing the Non categorical features
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
standardized_df = pd.DataFrame(ss.fit_transform(dataset[num_features]), columns = num_features)
standardized_df['Exited'] = dataset['Exited']
standardized_df.head()


# In[302]:


# Encoding for categorical data
from sklearn.preprocessing import LabelEncoder
le_geography = LabelEncoder()
le_gender = LabelEncoder()
le_HasCrCard = LabelEncoder()
le_IsActiveMember = LabelEncoder()
le_NumOfProducts = LabelEncoder()
le_Tenure = LabelEncoder()

le_df = pd.DataFrame()

le_df['Geography'] = le_geography.fit_transform(dataset['Geography'])
le_df['Gender'] = le_gender.fit_transform(dataset['Gender'])
le_df['HasCrCard'] = le_HasCrCard.fit_transform(dataset['HasCrCard'])
le_df['IsActiveMember'] = le_IsActiveMember.fit_transform(dataset['IsActiveMember'])
le_df['NumOfProducts'] = le_NumOfProducts.fit_transform(dataset['NumOfProducts'])
le_df['Tenure'] = le_Tenure.fit_transform(dataset['Tenure'])

le_df.head()


# In[303]:


# Appending scalled attributes DF with encoded columns
model_df = pd.concat([standardized_df,le_df],axis=1)
model_df.head()


# In[305]:


# Exited column is our target attribute and rest are featurs
X = model_df.drop(['Exited'],axis=1)
y = model_df['Exited']


# In[306]:


model_df.head()


# In[307]:


# Splitting the dataset into the Training and Testing set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 7)


# In[308]:


print(f'training shapes: {X_train.shape}, {y_train.shape}')
print(f'testing shapes: {X_test.shape}, {y_test.shape}')

# Lets try decision tree and KNeighborsClassifier
# In[333]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report

dt_clf = DecisionTreeClassifier(max_depth=5,max_features='sqrt')

dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
print(classification_report(y_test, pred))


# In[331]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
pred = neigh.predict(X_test)
print(classification_report(y_test, pred))

Creating a NN  model:
Keras model object can be created with Sequential class

At the outset, the model is empty per se. It is completed by adding additional layers and compilation
# In[309]:


model1 = Sequential()

from tensorflow.keras.metrics import Recall

Adding layers [layers and activations]:


Keras layers can be added to the model

Adding layers are like stacking lego blocks one by one

It should be noted that as this is a classification problem, sigmoid layer (softmax for multi-class problems) should be added
# In[310]:


model1.add(Dense(32, input_shape = (10,), activation = 'relu'))
model1.add(Dense(16, activation = 'relu'))
model1.add(Dense(16, activation = 'relu'))
model1.add(Dense(1, activation = 'sigmoid'))

Model compile [optimizers and loss functions]
Keras model should be "compiled" prior to training

Types of loss (function) and optimizer should be designated
# In[311]:


sgd = optimizers.Adam(lr = 0.001)


# In[312]:


model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics=["accuracy"])

Summary of Model
# In[313]:


model1.summary()


# In[314]:


model1.fit(X_train, y_train.values, batch_size = 20, epochs = 50, verbose = 1)


# In[315]:


#predicting the results of model

y_pred = model1.predict(X_test)
y_pred = (y_pred > 0.5) #to classify each probability into True or False

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print (cm, '\n\n', y_pred[:5, :])


# In[325]:


#accuracy of model
print ((1500 + 201)/2000)


# In[317]:


#Model looks good and slightly overfit, Now lets adjust hyperparameter and see if can get better result for moodel2


# In[318]:


model2 = Sequential()
model2.add(Dense(32, input_shape = (10,), activation = 'relu'))
model2.add(Dense(16, activation = 'relu'))
model2.add(Dense(16, activation = 'relu'))
model2.add(Dense(1, activation = 'sigmoid'))


# In[319]:


sgd = optimizers.Adam(lr = 0.001)


# In[320]:


model2.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics=["accuracy"])


# In[321]:


model2.summary()


# In[322]:


model2.fit(X_train, y_train.values, batch_size = 200, epochs = 50, verbose = 2)


# In[323]:


#predicting the results using model1

y_pred = model2.predict(X_test)
y_pred = (y_pred > 0.5) #to classify each probability into True or False

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print (cm, '\n\n', y_pred[:5, :])


# In[326]:


#accuracy of model2
print ((1475 + 224)/2000)


# In[73]:


#We found this model2 better slightly than model1

