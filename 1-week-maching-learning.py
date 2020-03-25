#!/usr/bin/env python
# coding: utf-8

# # AMAN OJHA - 2017BTECHCSE021

# # PANDAS

# ### pandas series creation and indexing

# In[1]:


import pandas as pd
step_data = [3620, 7891, 9761,
3907, 4338, 5373]
step_counts = pd.Series(step_data,
name='steps')
print(step_counts)


# In[2]:


step_counts.index = pd.date_range('20150329',periods=6)
print(step_counts)


# In[3]:


print(step_counts['2015-04-01'])


# In[4]:


print(step_counts[3])


# In[5]:


print(step_counts['2015-04'])


# ### pandas data types and imputation

# In[6]:


import numpy as np
print(step_counts.dtypes)
step_counts = step_counts.astype(np.float)
print(step_counts.dtypes)


# In[7]:


step_counts[1:3] = np.NaN
step_counts = step_counts.fillna(0.)
print(step_counts[1:3])


# ### pandas dataframe creation and methods

# In[8]:


cycling_data = [10.7,0.0,None,2.4,15.3,10.9,0,None]
joined_data = list(zip(step_data,cycling_data))
activity_df = pd.DataFrame(joined_data)
print(activity_df)


# In[9]:


activity_df = pd.DataFrame(joined_data,index=pd.date_range('20150329',periods=6),columns=['walking','cycling'])
print(activity_df)


# ### indexing dataframe rows and columns

# In[10]:


print(activity_df.loc['2015-04-01'])


# In[11]:


print(activity_df.iloc[-3])


# In[12]:


print(activity_df['walking'])


# In[13]:


print(activity_df.walking)


# In[14]:


print(activity_df.iloc[:,0])


# ### reading data with pandas

# In[15]:


import pandas as pd
filepath = "C:/Users/vmpr9/Desktop/MachineLearning/Iris_Data.csv"
data = pd.read_csv(filepath)
print(data)


# ### assigning newdata to a dataframe

# In[16]:


data['sepal_area'] = data.sepal_length*data.sepal_width
print(data.iloc[:5,-3:])


# ### applying a function to a  dataframe column

# In[17]:


data['abbrev'] = (data.species.apply(lambda x:x.replace('Iris-','')))
print(data.iloc[:5,-3:])


# ### concatenating two  dataframe

# In[18]:


small_data = pd.concat([data.iloc[:2],data.iloc[-2:]])
print(small_data.iloc[:,-3:])


# ### aggregated statistics with groupby

# In[19]:


group_sizes = (data.groupby('species').size())
print(group_sizes)


# ### performing stastical  calculations

# In[20]:


print(data.mean())


# In[21]:


print(data.petal_length.median())


# In[22]:


print(data.petal_length.mode())


# In[23]:


print(data.petal_length.std())


# In[24]:


print(data.petal_length.var())


# In[25]:


print(data.petal_length.sem())


# In[26]:


print(data.describe())


# ### sampling from dataframe
# 

# In[32]:


sample = (data.sample(n=5,replace=False,random_state=42))
print(sample.iloc[:,-3:])


# # VISUALIZATION LIBRARIES

# ### basic scatter plot with matplotlib

# In[28]:


import matplotlib.pyplot as plt
plt.plot(data.sepal_length,data.sepal_width,ls='',marker='o')


# In[31]:


plt.plot(data.sepal_length,data.sepal_width,ls='',marker='o',label='sepal')
plt.plot(data.petal_length,data.petal_width,ls='',marker='o',label='petal')


# ### histographs with matplotlib

# In[33]:


plt.hist(data.sepal_length,bins=25)


# ### customizing matplotlib plots

# In[34]:


fig, ax = plt.subplots()
ax.barh(np.arange(10),data.sepal_width.iloc[:10])
ax.set_yticks(np.arange(0.4,10.4,1.0))
ax.set_yticklabels(np.arange(1,11))
ax.set(xlabel='xlabel',ylabel='ylabel',title='Title')


# ### incorporating statistical calculations

# In[35]:


(data.groupby('species').mean().plot(color=['red','blue','black','green'],fontsize=10.0,figsize=(4,4)))


# ### statistical plotting with seaborn

# In[37]:


import seaborn as sns
sns.jointplot(x='sepal_length',y='sepal_width',data=data,height=4)


# In[38]:


sns.pairplot(data,hue='species',height=3)

