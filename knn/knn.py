#!/usr/bin/env python
# coding: utf-8

# In[43]:


from __future__ import division
import struct
import gzip
import numpy as np
import pandas as pd
from math import sqrt
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from tqdm import tqdm_notebook as tqdm


# In[44]:


def read_idx(filename):
    with gzip.open(filename) as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


# ###### Declare our dataset arrays

# In[ ]:


# Create a numpy array for the training data from the mnist dataset 
raw_train = read_idx(r'train-images-idx3-ubyte.gz')
## Flatten the training array
train_data = np.reshape(raw_train, (60000, 28 * 28))
train_label = read_idx(r'train-labels-idx1-ubyte.gz')

# Create a numpy array for the test data from the mnist dataset
raw_test = read_idx(r't10k-images-idx3-ubyte.gz')
## Flatten the test array
test_data = np.reshape(raw_test, (10000, 28 * 28))
test_label = read_idx(r't10k-labels-idx1-ubyte.gz')


# In[45]:


train_label


# In[46]:


n_train = 6000
n_test = 1000
split_loc = 60000 # train and test split at location of 60k

X_train, y_train =train_data,train_label
X_test, y_test = test_data,test_label


# In[47]:


# converting the above numpy arrays to pandas dataframe
df_train = pd.DataFrame(X_train)
df_test = pd.DataFrame(X_test)


# In[48]:


# calculate euclidean distance
def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))


# In[49]:


# a list to store euclidean distance
train_distance_list = []
# a list to store index
train_ind_counter = []
# a list with all the K values
k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19,21]
# creating a list of list for storing the predictions for each value of K
train_pred_lists = [[] for _ in range(len(k_values))]

# training the knn model
# iterating through the training set
for i in range(0,6000):
    train_vec_one = df_train.iloc[i]
    # iterating through the training set
    for j in range(0,6000):
        train_vec = df_train.iloc[j]
        # calculate euclidean distance by calling function dist
        euclidean_dist = dist(train_vec_one,train_vec)
        train_distance_list.append(euclidean_dist)
        # increment the index
        train_ind_counter.append(j)
    
    # dictionary to store all the results
    d = {'index':train_ind_counter, 'distance': train_distance_list}
    # convert dictionary to dataframe
    df = pd.DataFrame(d, columns = ['index', 'distance'])
    # sort in ascending order by euclidean distance
    df_sorted = df.sort_values(by = 'distance')

    # iterate through each value of K
    for K in range(len(k_values)):
        index_list = list(df_sorted['index'][:k_values[K]])
        distance = list(df_sorted['distance'][:k_values[K]])
        res_list = [y_train[i] for i in index_list]
        # now get the count of the max class in result list
        pred_value = max(res_list,key=res_list.count)
        # storing every prediction for K in respective list
        train_pred_lists[K].append(pred_value)
    
    # reinitialize the list
    train_ind_counter = []
    train_distance_list = []


# In[ ]:


# a list to store euclidean distance
test_distance_list = []
# a list to store index
test_ind_counter = []
# creating a list of list for storing the predictions for each value of K
test_pred_lists = [[] for _ in range(len(k_values))]

# testing the knn model
# iterating through the test set
for i in range(0,1000):
    test_vec = df_test.iloc[i]
    # iterating through the training set
    for j in range(0,6000): 
        train_vec = df_train.iloc[j]
        # calculate euclidean distance
        euclidean_dist = dist(test_vec,train_vec)
        test_distance_list.append(euclidean_dist)
        # increment the index
        test_ind_counter.append(j)
    
    # dictionary to store all the results
    d = {'index':test_ind_counter, 'distance': test_distance_list}
    # convert dictionary to dataframe
    df = pd.DataFrame(d, columns = ['index', 'distance'])
    # sort in ascending order by euclidean distance
    df_sorted = df.sort_values(by = 'distance')

    # iterate through each value of K
    for K in range(len(k_values)):
        index_list = list(df_sorted['index'][:k_values[K]])
        distance = list(df_sorted['distance'][:k_values[K]])
        res_list = [train_pred_lists[K][ind] for ind in index_list]
        # now get the count of the max class in result list
        pred_value = max(res_list,key=res_list.count)
        # storing every prediction in respective list
        test_pred_lists[K].append(pred_value)
        
    # # reinitialize the list   
    test_ind_counter = []
    test_distance_list = []


# In[ ]:


# calculating results for train set
train_pred = 0
train_pred_result = []
for K in range(len(k_values)):
    # element wise comparison to find the accuracy
    for l1,l2 in zip(train_pred_lists[K], y_train.tolist()):
        if l1 == l2:
            # increment when there is a match
            train_pred += 1
    accuracy = train_pred/6000
    train_pred_result.append((round(accuracy*100,2)))
    print('The train accuracy is '+str(round(accuracy*100,2))+'% for K='+str(k_values[K]))
    train_pred = 0


# In[ ]:


# calculating results for test set
test_pred = 0
test_pred_result = []
for K in range(len(k_values)):
    for l1,l2 in zip(test_pred_lists[K], y_test.tolist()):
        if l1 == l2:
            test_pred += 1
    accuracy = test_pred/1000
    test_pred_result.append((round(accuracy*100,2)))
    print('The test accuracy is '+str(accuracy*100)+'% for K='+str(k_values[K]))
    test_pred = 0


# In[ ]:


# getting all the results for train and test in a dataframe
df_result = pd.DataFrame()
df_result['K value'] = k_values
df_result['train pred'] = train_pred_result
df_result['test pred'] = test_pred_result
df_result


# In[ ]:


plt.plot(df_result['K value'], df_result['train pred'], 'r', label = 'train pred')
plt.plot(df_result['K value'], df_result['test pred'], 'g', label = 'test pred')
plt.legend(loc='upper right')
plt.xlabel('K value')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy for train and test set')
plt.show()



