#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing numpy and pandas which are required for data-preprocessing

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # Loading the data

# In[2]:


df=pd.read_csv("datathon_train.csv")


# In[3]:


df1=pd.read_csv("testing_data.csv",header=None)


# In[4]:


df1


# In[5]:


df1


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.isnull().sum()


# In[9]:


df.info()


# In[10]:


df=df.dropna()


# In[11]:


df


# In[12]:


df['urls_name']=df['urls_name'].apply(lambda x:x.replace("://"," "))
df['urls_name']=df['urls_name'].apply(lambda x:x.replace("."," "))
df['urls_name']=df['urls_name'].apply(lambda x:x.replace("/"," "))
df['urls_name']=df['urls_name'].apply(lambda x:x.replace("?"," "))
df['urls_name']=df['urls_name'].apply(lambda x:x.replace("%20"," "))
df['urls_name']=df['urls_name'].apply(lambda x:x.replace("-"," "))


# In[13]:


df1.iloc[:,0] = df1.iloc[:,0].apply(lambda x:x.replace("://"," "))
df1.iloc[:,0] = df1.iloc[:,0].apply(lambda x:x.replace("."," "))
df1.iloc[:,0] = df1.iloc[:,0].apply(lambda x:x.replace("/"," "))
df1.iloc[:,0] = df1.iloc[:,0].apply(lambda x:x.replace("?"," "))
df1.iloc[:,0] = df1.iloc[:,0].apply(lambda x:x.replace("%20"," "))
df1.iloc[:,0] = df1.iloc[:,0].apply(lambda x:x.replace("-"," "))


# In[14]:


df1


# # CountVectorizer

# In[15]:


from sklearn.feature_extraction.text import CountVectorizer


# In[16]:


vect=CountVectorizer()
vector=vect.fit_transform(df['urls_name'])


# In[17]:


vector


# In[18]:


vector.shape


# # TfidfVectorizer

# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[20]:


vect1=TfidfVectorizer
vector1=vect.fit_transform(df['urls_name'])


# In[21]:


vector1.shape


# In[22]:


vector1


# # Splitting the data

# In[23]:


y=df['0']


# In[24]:


y


# In[25]:


x=vector


# In[26]:


x


# In[27]:


from sklearn.model_selection import KFold
import numpy as np

k_fold = KFold(n_splits=5)
for indices_train, indices_test in k_fold.split(x):
    print(indices_train, indices_test)


# # Training of data

# In[28]:


from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.2)


# In[29]:


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
def train(model, X, y):
    # train the model
    x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    
    # predict the training set
    pred = model.predict(x_test)
    print("accuracy",model.score(x_test,y_test))


# # Logistic Regression

# In[30]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
train(model,x,y)


# # MultinomialNB

# In[31]:


from sklearn.naive_bayes import MultinomialNB
model2 = MultinomialNB()
train(model2,x,y)


# # Saving in pickle

# In[32]:


import pickle
filename="deep.pkl"
pickle.dump(model,open(filename,'wb'))


# In[33]:


load_model=pickle.load(open(filename,'rb'))


# In[34]:


vector=vect.transform(df1.iloc[:,0])


# In[35]:


y_pred = load_model.predict(vector)


# In[36]:


test_df = pd.DataFrame()


# In[37]:


test_df['a'] = y_pred


# In[39]:


test_df.to_csv(r'C:\Users\jyoth\Desktop\model\res8.csv', index = False)


# In[40]:


test_df


# # XGBClassifier

# In[41]:


from xgboost import XGBClassifier


# In[42]:


model1=XGBClassifier()
train(model1,x,y)


# In[43]:


filename="deep1.pkl"
pickle.dump(model1,open(filename,'wb'))


# In[44]:


load_model1=pickle.load(open(filename,'rb'))


# In[45]:


vector=vect.transform(df1.iloc[:,0])


# In[46]:


y_pred = load_model1.predict(vector)


# In[47]:


test_df1 = pd.DataFrame()


# In[48]:


test_df1['a'] = y_pred


# In[49]:


test_df1.to_csv(r'C:\Users\jyoth\Desktop\model\res5.csv', index = False)


# In[50]:


filename="deep2.pkl"
pickle.dump(model2,open(filename,'wb'))


# In[51]:


load_model2=pickle.load(open(filename,'rb'))


# In[52]:


vector=vect.transform(df1.iloc[:,0])


# In[53]:


y_pred = load_model2.predict(vector)


# In[54]:


test_df2 = pd.DataFrame()


# In[55]:


test_df2['a'] = y_pred


# In[56]:


test_df2.to_csv(r'C:\Users\jyoth\Desktop\model\res6.csv', index = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




