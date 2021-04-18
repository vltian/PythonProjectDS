#!/usr/bin/env python
# coding: utf-8

# ### Задание 1

# Импортируйте библиотеки pandas, numpy и matplotlib.
# 
# Загрузите "Boston House Prices dataset" из встроенных наборов 
# данных библиотеки sklearn.
# 
# Создайте датафреймы X и y из этих данных.
# 
# Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test)
# с помощью функции train_test_split так, чтобы размер тестовой выборки
# составлял 20% от всех данных, при этом аргумент random_state должен быть равен 42.
# 
# Масштабируйте данные с помощью StandardScaler.
# 
# Постройте модель TSNE на тренировочный данных с параметрами:
# n_components=2, learning_rate=250, random_state=42.
# 
# Постройте диаграмму рассеяния на этих данных.

# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

boston = load_boston()

X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns=["price"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# In[77]:


tsne = TSNE(n_components=2, learning_rate=250, random_state=42)

X_train_tsne = tsne.fit_transform(X_train)


# In[78]:


plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1])

plt.show()


# ### Задание 2
# 

# С помощью KMeans разбейте данные из тренировочного набора на 3 кластера,
# используйте все признаки из датафрейма X_train.
# 
# Параметр max_iter должен быть равен 100, random_state сделайте равным 42.
# 
# Постройте еще раз диаграмму рассеяния на данных, полученных с помощью TSNE,
# и раскрасьте точки из разных кластеров разными цветами.
# 
# Вычислите средние значения price и CRIM в разных кластерах.
# 

# In[104]:


from sklearn.cluster import KMeans

KM = KMeans(n_clusters=3, max_iter=100, random_state=42)

train_labels = KM.fit_predict(X_train)


# In[105]:


plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=train_labels)

plt.show()


# In[133]:


print("c0: avg CRIM -", X_train.CRIM[train_labels == 0].mean(),"avg", y_train[train_labels == 0].mean())


# In[134]:


print("c1: avg CRIM -", X_train.CRIM[train_labels == 1].mean(),"avg", y_train[train_labels == 1].mean())


# In[135]:


print("c2: avg CRIM -", X_train.CRIM[train_labels == 2].mean(),"avg", y_train[train_labels == 2].mean())


# ### *Задание 3

# Примените модель KMeans, построенную в предыдущем задании,
# к данным из тестового набора.
# 
# Вычислите средние значения price и CRIM в разных кластерах на тестовых данных.
# 

# In[89]:


test_labels = KM.fit_predict(X_test)


# In[129]:


print("c0: avg CRIM -", X_test.CRIM[test_labels == 0].mean(),"avg", y_test[test_labels == 0].mean())


# In[130]:


print("c1: avg CRIM -", X_test.CRIM[test_labels == 1].mean(),"avg", y_test[test_labels == 1].mean())


# In[131]:


print("c2: avg CRIM -", X_test.CRIM[test_labels == 2].mean(),"avg", y_test[test_labels == 2].mean())

