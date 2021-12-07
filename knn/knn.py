import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct
import gzip
from __future__ import division

def calcEuclidDist(x1,x2):   
    return np.sqrt(np.sum((x1-x2)**2))


def read_idx(filename):
    with gzip.open(filename) as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


k_val_input = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
data_train_raw = read_idx(r'train-images-idx3-ubyte.gz')
data_train = np.reshape(data_train_raw, (60000, 28 * 28))
train_label = read_idx(r'train-labels-idx1-ubyte.gz')
data_test_raw = read_idx(r't10k-images-idx3-ubyte.gz')
data_test = np.reshape(data_test_raw, (10000, 28 * 28))
test_label = read_idx(r't10k-labels-idx1-ubyte.gz')

X_train, y_train =data_train,train_label
X_test, y_test = data_test,test_label
df_train = pd.DataFrame(X_train)
df_test = pd.DataFrame(X_test)

train_data_distances = []
train_idx_list = []
train_data_pred = [[] for _ in range(len(k_val_input))]

test_data_distances = []
test_idx_list = []
test_data_pred = [[] for _ in range(len(k_val_input))]

for i in range(0,6000):
    vector1_training = df_train.iloc[i]
    for j in range(0,6000):
        temp_vector_training = df_train.iloc[j]
        dista = calcEuclidDist(vector1_training,temp_vector_training)
        train_data_distances.append(dista)
        train_idx_list.append(j)
    
    temp_dist = {'index':train_idx_list, 'distance': train_data_distances}
    df = pd.DataFrame(temp_dist, columns = ['index', 'distance'])
    sorted_df = df.sort_values(by = 'distance')

    for i in range(len(k_val_input)):
        index_list = list(sorted_df['index'][:k_val_input[i]])
        distance = list(sorted_df['distance'][:k_val_input[i]])
        res_list = [y_train[i] for i in index_list]
        pred_value = max(res_list,key=res_list.count)
        train_data_pred[i].append(pred_value)
    train_idx_list = []
    train_data_distances = []

for i in range(0,1000):
    test_vec = df_test.iloc[i]
    for j in range(0,6000): 
        temp_vector_test = df_train.iloc[j]
        dista = calcEuclidDist(test_vec,temp_vector_test)
        test_data_distances.append(dista)
        test_idx_list.append(j)
    
    temp_dist = {'index':test_idx_list, 'distance': test_data_distances}
    df = pd.DataFrame(temp_dist, columns = ['index', 'distance'])
    sorted_df = df.sort_values(by = 'distance')

    for i in range(len(k_val_input)):
        index_list = list(sorted_df['index'][:k_val_input[i]])
        distance = list(sorted_df['distance'][:k_val_input[i]])
        res_list = [train_data_pred[i][ind] for ind in index_list]
        pred_value = max(res_list,key=res_list.count)
        test_data_pred[i].append(pred_value)
          
    test_idx_list = []
    test_data_distances = []

train_pred = 0
train_pred_result = []
for i in range(len(k_val_input)):
    for l1,l2 in zip(train_data_pred[i], y_train.tolist()):
        if l1 == l2:
            train_pred += 1
    accuracy = train_pred/6000
    train_pred_result.append((round(accuracy*100,2)))
    print('The train accuracy is '+str(round(accuracy*100,2))+'% for K='+str(k_val_input[i]))
    train_pred = 0

test_pred = 0
test_pred_result = []
for i in range(len(k_val_input)):
    for l1,l2 in zip(test_data_pred[i], y_test.tolist()):
        if l1 == l2:
            test_pred += 1
    accuracy = test_pred/1000
    test_pred_result.append((round(accuracy*100,2)))
    print('The test accuracy is '+str(accuracy*100)+'% for K='+str(k_val_input[i]))
    test_pred = 0

df_result = pd.DataFrame()
df_result['K value'] = k_val_input
df_result['train pred'] = train_pred_result
df_result['test pred'] = test_pred_result
df_result

plt.plot(df_result['K value'], df_result['train pred'], 'g', label = 'train pred')
plt.plot(df_result['K value'], df_result['test pred'], 'r', label = 'test pred')
plt.show()
