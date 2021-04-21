#!/usr/bin/env python
# coding: utf-8

# In[1]:


# импорт библиотек
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score as r2, accuracy_score
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from matplotlib import pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[2]:


DATASET_TEST_PATH = './test.csv'
DATASET_TRAIN_PATH = './train.csv'


# In[3]:


df_train = pd.read_csv(DATASET_TRAIN_PATH, sep=',')
df_train.head()


# ### Поиск пропусков

# In[4]:


df_train.isna().sum()


# ### Определение категориальных признаков

# In[5]:


df_train.nunique()


# In[6]:


# подготовка категориальных признаков
df_train.replace({'Ecology_2': {'A':0, 'B':1}, 'Ecology_3': {'A':0, 'B':1}, 'Shops_2': {'A':0, 'B':1}}, inplace = True)


# In[7]:


# разделение тренировочного датасета на тренировочный и валидационный
X_train, X_test, y_train, y_test = train_test_split(df_train.set_index('Id').drop(['LifeSquare', 'Healthcare_1', 'Price'], 
                       axis = 'columns'), df_train['Price'], test_size=0.2, random_state=21)


# In[8]:


# выбор и обучение модели, расчет коэф-та r2 для тренировочной и валидационной моделей
RFR_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=26)
RFR_model.fit(X_train, y_train)
predict_train_1 = RFR_model.predict(X_train)
predict_valid_1 = RFR_model.predict(X_test)
print('r2_train = {r2}'.format(r2 = r2(y_train, predict_train_1)))
print('r2_valid = {r2}'.format(r2 = r2(y_test, predict_valid_1)))


# ### Тестовый датасет

# In[9]:


df_test = pd.read_csv(DATASET_TEST_PATH, sep=',')
df_test.head(10)


# In[10]:


df_test.shape


# In[11]:


df_test.isna().sum()


# In[12]:


# подготовка категориальных признаков
df_test.replace({'Ecology_2': {'A':0, 'B':1}, 'Ecology_3': {'A':0, 'B':1}, 'Shops_2': {'A':0, 'B':1}}, inplace = True)


# In[13]:


# расчет целевого показателя
predict_test_1 = RFR_model.predict(df_test.set_index('Id').drop(['LifeSquare', 'Healthcare_1'], axis = 'columns'))


# In[14]:


df_test['Price'] = predict_test_1
df_test.head()


# ###  Добавление признаков

# In[15]:


# Средняя цена за м2 по району
av_price_distr = df_train.groupby('DistrictId')[['Price', 'Square']].sum().reset_index()
av_price_distr['sqm_price'] = av_price_distr['Price'] / av_price_distr['Square']
av_price_distr = av_price_distr[['DistrictId', 'sqm_price']]
av_price_distr


# In[16]:


df_train = pd.merge(df_train, av_price_distr, how = 'left', on = 'DistrictId')
df_train


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(df_train.set_index('Id').drop(['LifeSquare', 'Healthcare_1', 'Price', 'DistrictId'], 
                       axis = 'columns'), df_train['Price'], test_size=0.2, random_state=21)


# In[18]:


RFR_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=26)
RFR_model.fit(X_train, y_train)
predict_train_1 = RFR_model.predict(X_train)
predict_valid_1 = RFR_model.predict(X_test)
print('r2_train = {r2}'.format(r2 = r2(y_train, predict_train_1)))
print('r2_valid = {r2}'.format(r2 = r2(y_test, predict_valid_1)))


# In[19]:


df_train.info()


# In[20]:


df_test.drop(['Price'], axis='columns', inplace = True)


# In[21]:


df_test = pd.merge(df_test, av_price_distr, how = 'left', on = 'DistrictId')
df_test.loc[df_test['sqm_price'].isnull(), 'sqm_price'] = df_test['sqm_price'].median()
df_test.info()


# In[22]:


predict_test_2 = RFR_model.predict(df_test.set_index('Id').drop(['LifeSquare', 'Healthcare_1', 'DistrictId'], axis = 'columns'))


# In[23]:


df_test['Price'] = predict_test_2
df_test.head()


# In[24]:


df_test[['Id', 'Price']].to_csv('Vtyan_predictions.csv', index=False)

