#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

# In[2]:

DATASET_PATH = './training_project_data.csv'
#PREP_DATASET_PATH = './data/training_project_data_prep.csv'

# In[3]:

df = pd.read_csv(DATASET_PATH)
df.head()

# In[4]:

X_train, X_valid, y_train, y_valid = train_test_split(df.drop(['NEXT_MONTH_DEFAULT'], 
                       axis = 'columns'), df['NEXT_MONTH_DEFAULT'], test_size=0.2, random_state=21)

# In[5]:

scaler = MinMaxScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)


# In[6]:


clf = SVC(gamma="auto")

clf.fit(X_train, y_train)

y_pred = clf.predict(X_valid)
y_pred_train = clf.predict(X_train)

cm = confusion_matrix(y_valid, y_pred)


# In[7]:


from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

pc = precision_score(y_valid, y_pred)
pc


# In[8]:


rc = recall_score(y_valid, y_pred)
rc


# In[9]:


avg = (pc + rc)/2
avg


# In[10]:


from sklearn.metrics import f1_score

fm = f1_score(y_valid, y_pred)
fm


# ## Домашнее задание

# Приведите по два примера, когда лучше максимизировать Precision, а когда Recall.
# 
# Почему мы используем F-меру? Почему, например, нельзя просто взять среднее от Precision и Recall?

# 1. Precision - когда критична ошибочная классификация как представителя класса, например
# - профилирование покупателей
# - идентификация как благонадежного заемщика
# Recall - когда критично пропустить представителя класса, например
# - идентификация террориста
# - выявление злокачественной опухоли
# 
# 2. Показатели Precision  и Recall "разнонаправлены", что отражается в разных знаменателях, то есть у них разный "масштаб".
# 
# F1=2⋅(precision⋅recall)/(precision+recall) = 2TP / ((TP+FN) + (TP+FP)) = TP / ср.арифм знаменателей
# 
# Чтобы вывести усредненный показатель, нужно усреднить знаменатели.
