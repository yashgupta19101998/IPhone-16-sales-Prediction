#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.request import urlopen


# In[2]:


api_key = 'd6b95b5ddeba46ecaba667ddf118cd22'

#Define the endpoint and parameters

url = 'https://newsapi.org/v2/everything'

params = {

    'q': 'iphone16',  # Search query for articles about "social media"

    'apiKey': api_key,

    'language': 'en',  # Only return articles in English

    'pageSize': 10  # Limit to 10 results

}


# In[3]:


#API

import requests

response = requests.get(url, params=params)

print(response.json())

# Step 4: Parse the response JSON

data = response.json()


# In[4]:


print(data)


# In[5]:


df1 = pd.read_excel(r"E:\Iphone\iPhone_16_Market_Analysis_1000_Rows.xlsx")
df1


# In[6]:


df1.info()


# In[7]:


# Catagorical Columns
list_cate = []

for i in list(df1.columns):
    if df1[i].dtype == 'O': # O reffers to object
        print(i)
        list_cate.append(i)


# In[8]:


#Numerical columns
set(list_cate) ^ set(list(df1.columns))


# In[9]:


df1['Months'] = df1['Date'].dt.month
df1.tail()


# In[10]:


df2 = pd.read_excel(r"E:\Iphone\iPhone_16_Technology_Trends_Analysis_1000_Rows.xlsx")
df2


# In[11]:


df2.info()


# In[12]:


# Catagorical Columns
list_cate = []

for i in list(df2.columns):
    if df2[i].dtype == 'O': # O reffers to object
        print(i)
        list_cate.append(i)


# In[13]:


#Numerical columns
set(list_cate) ^ set(list(df2.columns))


# In[14]:


# Now deciding which columns are required for analyzing the future of Iphone in Indian Market from 
# from df1 & df2


# In[15]:


df1_num = df1[['Competitor_Feature_Rating', 'Date', 'Market_Share', 'Marketing_Spend_Competitor', 'Marketing_Spend_iPhone_16',
 'Price', 'Sales_Volume', 'User_Rating', 'iPhone_16_Feature_Rating', 'iPhone_16_Price', 'iPhone_16_Project_Market_Share',
 'iPhone_16_Projected_Sales', 'iPhone_16_User_Rating']]
df1_num


# In[16]:


df1_num['Months'] = df1_num['Date'].dt.month
df1_num.tail()


# In[17]:


sns.lineplot(data = df1_num, x = 'iPhone_16_Feature_Rating', y = 'iPhone_16_Price')


# In[18]:


sns.lineplot(data = df1_num, x = 'iPhone_16_User_Rating', y = 'Sales_Volume')


# In[19]:


sns.lineplot(data = df1_num, x = 'iPhone_16_User_Rating', y = 'iPhone_16_Price')


# In[20]:


sns.catplot(data= df1, x = 'Sales_Volume', y = 'iPhone_16_Promotion_Type',hue = 'Feature_Comparison', kind = 'bar')


# In[21]:


sns.catplot(data = df1, x = 'Sales_Volume',y = 'iPhone_16_Promotion_Type', kind = 'bar')


# In[22]:


sns.catplot(data = df1, x = 'Sales_Volume',y = 'Feature_Comparison', kind = 'bar')


# In[23]:


sns.catplot(data = df1_num, x = 'Months',y = 'Sales_Volume', kind = 'bar')


# In[24]:


sns.catplot(data = df1_num, x = 'Months',y = 'Price', kind = 'bar')


# In[25]:


df2['Months'] = df2['Date'].dt.month
df2.tail()


# In[53]:


df2.info()


# In[54]:


sns.catplot(data = df2, y = 'Technology_Trend',x = 'Trend_Influence_on_Sales', kind = 'bar')


# In[55]:


sns.catplot(data = df2, y = 'iPhone_16_Feature',x = 'Trend_Influence_on_Sales', kind = 'bar')


# In[57]:


sns.catplot(data = df2, y = 'Technology_Trend',x = 'Competitor_Adoption_Rate', kind = 'bar')


# ## Train and Test model

# In[29]:


#Modeling
# The price column goes in y
y = df1_num['Price'] 


# Everything apart from price column goes in x
x = df1_num.drop(['Price','Date'],axis=1)


# In[30]:


y.head()


# In[31]:


x.head()


# In[32]:


# Import train-test scikit learn
from sklearn.model_selection import train_test_split


# In[33]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0, test_size=0.2)


# In[34]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ##Fitting The Model

# In[35]:


# Importing Linear Regression model from scikit learn
from sklearn.linear_model import LinearRegression


# In[36]:


# Initialising my model
lr = LinearRegression()


# In[37]:


# Fitting my model
lr.fit(x_train,y_train)


# In[38]:


# Predicting the Salary for the Test values
y_pred = lr.predict(x_test)


# In[39]:


y_pred[:10]


# In[40]:


y_test[:10]


# #Now looking at how good my predictions are

# In[41]:


## Looking at the score


# In[42]:


# Importing metrics for the evaluation of the model
from sklearn.metrics import r2_score,mean_squared_error


# In[43]:


# calculate Mean square error
mean_squared_error(y_test,y_pred)


# In[44]:


mse = mean_squared_error(y_test,y_pred)
mse


# In[45]:


# Calculate R square vale
r2_score(y_test,y_pred)


# In[50]:


df1_num['Price'].mean()


# In[51]:


x_range = [i for i in range(len(y_test))]
x_range


# In[52]:


import matplotlib.pyplot as plt


temp = [6.9727 for i in range(200)]

plt.xlabel('Sales_Volume')
plt.ylabel('Price')
plt.title('Prediction')

plt.plot(x_range,y_test)
plt.plot(x_range,y_pred)

plt.plot(x_range,temp)

# Function add a legend  
plt.legend(["Real", "Predicted",'Mean'])


# # Hence we can impute that Iphone can look up to inhance their sales in future by:
# ## 1.Giving aexchange offer on sales
# ## 2.Focusing more on camera and higher refresh rate at the timer of product launch
# ## 3. It is also observed that Iphone tends to increase its price more when its user rating increases due to which its sales drops
# ## 4. It is observed that people tends to buy more in India at the price range of 70000 to 80000. So SKU on that models production to be increased.
# ## 5. In case of product adoptation and liking of customer Iphone can work on AI based photography, 5G adaptablity and on foaldable screen technology to attract more customers

# In[ ]:




