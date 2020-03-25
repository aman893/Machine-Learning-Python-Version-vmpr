#!/usr/bin/env python
# coding: utf-8

# # SUPERVISED LEARNING : REGRESSION AND CLASSIFIER

# ## METHODS : K NEAREST NEIGHBORS

# ### * K-nearest neighbors requires scaled data i.e, Feauture Scaling

# #### **Feauture Scaling : @ Standard Scaler -> mean data and scale to unit variance; @ Minimum-Maximum Scaler -> scale data to fixed range; @ Maximum Absolute Value Scaler -> scale maximum absolute value 
#                        

# ### Feature Scaling : The Syntax

# In[10]:


#from sklearn.preprocessing import StandardScaler
#StdSc = StandardScaler()
#StdSc = StdSc.fit(X_data)
#X_scaled = KNN.transform(X_data)


# In[11]:


#from sklearn.preprocessing import MinMaxScaler

#msc = MinMaxScaler()

#data = pd.DataFrame(msc.fit_transform(data),columns=data.columns) # this is an np.array, not a dataframe.


# ## K Nearest Neighbors : The Syntax

# ### Classification Method

# In[15]:


#from sklearn.neighbors import KNeighborsClassifier 
#KNN = KNeighborsClassifier(n_neighbors=3)
#KNN = KNN.fit(X_data,y_data)
#y_predict = KNN.predict(X_data)


# ### Regression Method

# In[16]:


#from sklearn.neighbors import KNeighborsRegressor
#KNN = KNeighborsRegressor(n_neighbors=3)
#KNN = KNN.fit(X_data,y_data)
#y_predict = KNN.predict(X_data)


# ### Alternatiovely, Full explanation for importing the class in any method like for Regression

# In[17]:


#class sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)


# ### algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
# Algorithm used to compute the nearest neighbors:
# 
# ‘ball_tree’ will use BallTree
# 
# ‘kd_tree’ will use KDTree
# 
# ‘brute’ will use a brute-force search.
# 
# ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
