#!/usr/bin/env python
# coding: utf-8

# ### Задание 1
# Импортируйте библиотеку Pandas и дайте ей псевдоним pd. Создайте датафрейм authors со столбцами author_id и author_name, в которых соответственно содержатся данные: [1, 2, 3] и ['Тургенев', 'Чехов', 'Островский'].
# Затем создайте датафрейм book cо столбцами author_id, book_title и price, в которых соответственно содержатся данные:  
# [1, 1, 1, 2, 2, 3, 3],
# ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
# [450, 300, 350, 500, 450, 370, 290].

# In[1]:


import pandas as pd


# In[127]:


authors = pd.DataFrame({'author_id': [1, 2, 3],
                       'author_name': ['Тургенев', 'Чехов', 'Островский']})
book = pd.DataFrame({'author_id': [1, 1, 1, 2, 2, 3, 3],
                    'book_title': ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
                    'price': [450, 300, 350, 500, 450, 370, 290]})
authors


# In[128]:


book


# ### Задание 2
# Получите датафрейм authors_price, соединив датафреймы authors и books по полю author_id.

# In[11]:


authors_price = pd.merge(authors, book, on = 'author_id', how='inner')
authors_price


# ### Задание 3
# Создайте датафрейм top5, в котором содержатся строки из authors_price с пятью самыми дорогими книгами.
# 

# In[32]:


list_desc = authors_price.sort_values(['price'], ascending = [False])
top5 = list_desc.head(5)
top5


# In[35]:


top5 = authors_price.nlargest(5, 'price')
top5


# ### Задание 4
# Создайте датафрейм authors_stat на основе информации из authors_price. В датафрейме authors_stat должны быть четыре столбца:
# author_name, min_price, max_price и mean_price,
# в которых должны содержаться соответственно имя автора, минимальная, максимальная и средняя цена на книги этого автора.

# In[120]:


group_aut = authors_price.groupby("author_name").agg({"price": ['min', 'max', 'mean']})
group_aut


# ### Задание 5
# Создайте новый столбец в датафрейме authors_price под названием cover, в нем будут располагаться данные о том, какая обложка у данной книги - твердая или мягкая. В этот столбец поместите данные из следующего списка:
# ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая'].
# 
# Просмотрите документацию по функции pd.pivot_table с помощью вопросительного знака.Для каждого автора посчитайте суммарную стоимость книг в твердой и мягкой обложке. Используйте для этого функцию pd.pivot_table. 
# При этом столбцы должны называться "твердая" и "мягкая", а индексами должны быть фамилии авторов. Пропущенные значения стоимостей заполните нулями, при необходимости загрузите библиотеку Numpy.
# 
# Назовите полученный датасет book_info и сохраните его в формат pickle под названием "book_info.pkl". Затем загрузите из этого файла датафрейм и назовите его book_info2. Удостоверьтесь, что датафреймы book_info и book_info2 идентичны.

# In[145]:


authors_price.loc [:, 'cover'] =  ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая']
authors_price


# In[149]:


get_ipython().run_line_magic('pinfo', 'pd.pivot_table')


# In[152]:


import numpy as np
book_info = pd.pivot_table(authors_price, values='price', index=['author_name'], columns=['cover'], aggfunc=np.sum, fill_value=0)
book_info


# In[155]:


book_info.to_pickle("book_info.pkl")


# In[157]:


book_info2 = pd.read_pickle("book_info.pkl")
book_info2

