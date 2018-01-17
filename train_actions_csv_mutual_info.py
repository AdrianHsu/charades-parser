
# coding: utf-8

# In[87]:


import os
import pandas as pd
import numpy as np
import sklearn
import csv


# In[88]:


df = pd.read_csv('Charades_v1_train.csv')


# In[89]:


df.head()


# In[90]:


print(df.shape)
df = df[pd.notnull(df['actions'])] # drop NA in column['actions']
print(df.shape)

# df = df[df.quality >=6 ]
# df = df[df.relevance == 7] #  和 script 的相關性，但我們不需要管 script ，所以用不到
df['origin_index'] = df.index.values
df.head()


# In[91]:


df = df.reset_index(drop=True)
df.head()


# In[92]:


dat_id = df['id']
dat_actions = df['actions']
dat_origin_index = df['origin_index']
dat = pd.concat([dat_id, dat_origin_index, dat_actions], axis=1)
split_arr = dat['actions'].str.split(';')


# In[93]:


large_arr = []
for element in enumerate(split_arr):
    arr = []
    time = []
    for i in element[1]:
        k = i.split(" ")
        d = k[0]
        t = k[1]
        arr.append(d)
        time.append(t)
#     print(arr)
#     print(time)
    arr = [x for _,x in sorted(zip(time,arr))]
#     print(arr)
    large_arr.append(arr)
dat['split'] = large_arr


# In[94]:


dat = dat.drop('actions', 1)


# In[95]:


str_actions = dat['split']
int_actions = []
for action in str_actions:
    arr = []
    for act in action:
        a = int(act[1:])
        arr.append(a)
    int_actions.append(arr)


# In[96]:


dat
# str_actions


# In[97]:


join_str = []
for it in str_actions:
    for s in it:
        join_str.append(s)
        
# join_str


# In[98]:


import nltk
from nltk.corpus import brown
from nltk import WordNetLemmatizer
from math import log 

# brown = nltk.download('brown')
# wd = nltk.download('wordnet')

wnl=WordNetLemmatizer()
# lemmatize 詞型還原

_Fdist = nltk.FreqDist([wnl.lemmatize(w.lower()) for w in join_str])

_Sents = [[wnl.lemmatize(j.lower()) for j in i] for i in str_actions]

MAX_DISTANCE = 100000 # 只有這兩個動作距離 = 1，這筆 video 才算數。

def inSents(x, y):
    c = 0
    l = []
    for s in _Sents:
        
#         first method:
        index1 = 0
        index2 = 0
        bool1 = False
        bool2 = False
        k = 0
        for index, st in enumerate(s):    
            if x == st and not bool1:
                k += 1
                index1 = index
                bool1 = True
            if y == st and not bool2:
                k += 1
                index2 = index  
                bool2 = True
        if k == 2: # x in s and y in s
            if abs(index1 - index2) < MAX_DISTANCE: # i.e 必定為 1 
                l.append(s)
                
#         second method:
#         if x in s and y in s:
#             l.append(s)
        
    return l

# int_actions
def p(x):
    return _Fdist[x]/ float(sum(_Fdist.values())) #float(len(_Fdist))
    

def pxy(x,y):
    return (len(inSents(x,y)) + 1) / float(len(_Sents) )

def pmi(x,y):
    return  log(pxy(x,y)/(p(x)*p(y)),2) 


# In[99]:


_Fdist
# _Sents


# In[100]:


# c000 Holding some clothes
# c001 Putting clothes somewhere
print(p('c000'))
print(p('c001'))
print(pxy('c000', 'c001')) # 0.0405
print(pmi('c000', 'c001'))


# In[101]:


# c059 Sitting in a chair
# c011 Sitting at a table
print(p('c059'))
print(p('c011'))
print(pxy('c059', 'c011')) #0.0684
print(pmi('c059', 'c011'))


# In[102]:


# c057 Taking off some shoes
# c065 Eating a sandwich
print(p('c057'))
print(p('c065'))
print(pxy('c057', 'c065'))
print(pmi('c057', 'c065'))


# In[103]:


classes_raw = open('Charades_v1_classes.txt', 'r')
classes = []
for i in classes_raw.readlines():
    arr = i.split(' ', 1)
    classes.append(arr)
# classes[0][0]
# len(classes) # 157x2, c000 to c156


# In[104]:


two = []
two_pmi = []
for i in range(0, 157):
    for j in range(i+1, 157):
        arr = [classes[i], classes[j]]
        two.append(arr)
        two_pmi.append(pmi(classes[i][0], classes[j][0]))


# In[117]:


print(len(two_pmi))
two_dim = np.zeros((157, 157))
k = 0
for i in range(0, 157):
    for j in range(i+1, 157):
        two_dim[j][i] = two_pmi[k]
        k += 1
print(two_dim)
import matplotlib.pyplot as plt
# %matplotlib inline

arr = np.array(two_dim)
np.save('class_pmi', arr)

def myplot(arr):
    plt.imshow(arr, interpolation=None)
    plt.savefig('class.png', dpi=1000)
    plt.show()

myplot(arr)

