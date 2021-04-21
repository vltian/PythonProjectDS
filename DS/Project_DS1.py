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


# подготовка признаков
df_train.replace({'Rooms': {0 : 1}}, inplace = True)
df_train.replace({'Ecology_2': {'A':0, 'B':1}, 'Ecology_3': {'A':0, 'B':1}, 'Shops_2': {'A':0, 'B':1}}, inplace = True)


# In[7]:


# разделение тренировочного датасета на тренировочный и валидационный
X_train, X_valid, y_train, y_valid = train_test_split(df_train.set_index('Id').drop(['LifeSquare', 'Healthcare_1', 'Price'], 
                       axis = 'columns'), df_train['Price'], test_size=0.2, random_state=21)
X_valid.head()


# In[8]:


# стандартизация и обучение стандартизованных датасетов
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_valid_scaled = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)


# In[9]:


#  t-SNE алгоритм - понижение размерности
tsne = TSNE(n_components=2, learning_rate=500, random_state=60)
X_train_tsne = tsne.fit_transform(X_train_scaled)


# In[10]:


plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1])
plt.show()


# In[11]:


# кластеризация
kmeans = KMeans(n_clusters=2, random_state=42)

labels_train = kmeans.fit_predict(X_train_scaled)

plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=labels_train)

plt.show()


# In[12]:


centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=labels_train)
plt.scatter(centers[:, 0], centers[:, 1], marker='D', color='red')


# In[13]:


# выбор и обучение модели, расчет коэф-та r2 для тренировочной и валидационной моделей
RFR_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=26)
RFR_model0 = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=26)
RFR_model1 = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=26)
RFR_model.fit(X_train, y_train)
predict_train_1 = RFR_model.predict(X_train)
predict_valid_1 = RFR_model.predict(X_valid)
print('r2_train = {r2}'.format(r2 = r2(y_train, predict_train_1)))
print('r2_valid = {r2}'.format(r2 = r2(y_valid, predict_valid_1)))


# In[14]:


labels_test = kmeans.predict(X_valid_scaled)


# In[15]:


# Модель для кластера 0
RFR_model0.fit(X_train_scaled.loc[labels_train == 0], y_train[labels_train == 0])
y_valid_pred_0 = RFR_model0.predict(X_valid_scaled.loc[labels_test == 0])

r2(y_valid[labels_test == 0], y_valid_pred_0)


# In[16]:


# Модель для кластера 1
RFR_model1.fit(X_train_scaled.loc[labels_train == 1], y_train[labels_train == 1])
y_valid_pred_1 = RFR_model1.predict(X_valid_scaled.loc[labels_test == 1])

r2(y_valid[labels_test == 1], y_valid_pred_1)


# In[17]:


# Модель общая
y_valid_all = np.hstack([y_valid[labels_test == 0], y_valid[labels_test == 1]])
y_valid_pred_all = np.hstack([y_valid_pred_0, y_valid_pred_1])

r2(y_valid_all, y_valid_pred_all)


# ### Кластеризация дала небольшое улучшение метрики

# ### Тестовый датасет

# In[18]:


df_test = pd.read_csv(DATASET_TEST_PATH, sep=',')
#df_test.head(10)


# In[19]:


df_test.shape


# In[20]:


df_test.isna().sum()


# In[21]:


# подготовка признаков
df_test.replace({'Rooms': {0 : 1}}, inplace = True)
df_test.replace({'Ecology_2': {'A':0, 'B':1}, 'Ecology_3': {'A':0, 'B':1}, 'Shops_2': {'A':0, 'B':1}}, inplace = True)
df_test.drop(['LifeSquare', 'Healthcare_1'], axis = 'columns', inplace = True)
df_test.set_index('Id', inplace = True)


# In[22]:


df_test.info()


# In[23]:


#for colname in ['DistrictId', 'Floor', 'HouseYear', 'Ecology_2', 'Ecology_3', 'Social_1', 'Social_2', 'Social_3', 
                #'Helthcare_2', 'Shops_1', 'Shops_2']:
   # df_test[colname] = df_test[colname].astype('int64')


# In[24]:


df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)


# In[25]:


# расчет целевого показателя
predict_test_1 = RFR_model.predict(df_test_scaled)


# In[26]:


df_test['Price'] = predict_test_1
df_test.head()


# ###  Добавление признаков

# In[27]:


# Средняя цена за м2 по району
av_price_distr = df_train.groupby('DistrictId')[['Price', 'Square']].sum().reset_index()
av_price_distr['sqm_price'] = av_price_distr['Price'] / av_price_distr['Square']
av_price_distr = av_price_distr[['DistrictId', 'sqm_price']]
av_price_distr


# In[28]:


df_train = pd.merge(df_train, av_price_distr, how = 'left', on = 'DistrictId')
df_train


# In[29]:


X_train, X_valid, y_train, y_valid = train_test_split(df_train.set_index('Id').drop(['LifeSquare', 'Healthcare_1', 'DistrictId', 'Price'], 
                       axis = 'columns'), df_train['Price'], test_size=0.2, random_state=21)


# In[30]:


X_train.head()


# In[31]:


scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_valid_scaled = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)


# In[32]:


tsne = TSNE(n_components=2, learning_rate=500, random_state=60)
X_train_tsne = tsne.fit_transform(X_train_scaled)


# In[33]:


X_train_scaled.shape


# In[34]:


X_train_tsne.shape


# In[35]:


plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1])

plt.show()


# In[36]:


kmeans = KMeans(n_clusters=2, random_state=42)

labels_train = kmeans.fit_predict(X_train_scaled)

plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=labels_train)

plt.show()


# In[37]:


centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=labels_train)
plt.scatter(centers[:, 0], centers[:, 1], marker='D', color='red')


# In[38]:


RFR_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=26)
RFR_model0 = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=26)
RFR_model1 = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=26)
RFR_model.fit(X_train, y_train)
predict_train_1 = RFR_model.predict(X_train)
predict_valid_1 = RFR_model.predict(X_valid)
print('r2_train = {r2}'.format(r2 = r2(y_train, predict_train_1)))
print('r2_valid = {r2}'.format(r2 = r2(y_valid, predict_valid_1)))


# In[39]:


labels_test = kmeans.predict(X_valid_scaled)


# In[40]:


RFR_model0.fit(X_train_scaled.loc[labels_train == 0], y_train[labels_train == 0])
y_valid_pred_0 = RFR_model0.predict(X_valid_scaled.loc[labels_test == 0])

r2(y_valid[labels_test == 0], y_valid_pred_0)


# In[41]:


RFR_model1.fit(X_train_scaled.loc[labels_train == 1], y_train[labels_train == 1])
y_valid_pred_1 = RFR_model1.predict(X_valid_scaled.loc[labels_test == 1])

r2(y_valid[labels_test == 1], y_valid_pred_1)


# In[42]:


y_valid_all = np.hstack([y_valid[labels_test == 0], y_valid[labels_test == 1]])
y_valid_pred_all = np.hstack([y_valid_pred_0, y_valid_pred_1])

r2(y_valid_all, y_valid_pred_all)


# ### Кластеризация после добавления признаков не улучшила метрику

# In[43]:


df_test = df_test.reset_index().merge(av_price_distr, how = 'left', on = 'DistrictId').set_index('Id')
df_test.loc[df_test['sqm_price'].isnull(), 'sqm_price'] = df_test['sqm_price'].median()
df_test.info()


# In[44]:


df_test.drop(['Price','DistrictId'], axis = 'columns', inplace = True)
df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)


# In[45]:


predict_test_2 = RFR_model.predict(df_test_scaled)


# In[47]:


df_test['Price'] = predict_test_2
df_test.head()


# In[48]:


df_test.reset_index(inplace = True)


# In[49]:


df_test.info()


# In[50]:


df_test[['Id', 'Price']].to_csv('Vtyan_predictions1.csv', index=False)

