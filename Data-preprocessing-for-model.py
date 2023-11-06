#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values
print(X)
print(y)


# In[10]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


print(X)


# In[12]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],   remainder='passthrough') 
X = np.array(ct.fit_transform(X)) 
print(X) 


# In[13]:


from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
y = le.fit_transform(y)
print(y) 


# In[14]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  random_state = 1)  
print(X_train) 
print(X_test) 
print(y_train) 
print(y_test) 


# In[16]:


from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:]) 
X_test[:, 3:] = sc.transform(X_test[:, 3:]) 
print(X_train) 


# In[17]:


print(X_test)


# In[ ]:




