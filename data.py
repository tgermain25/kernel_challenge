import pandas as pd
import numpy as np

def train_set(X_path,Y_path, seq = ['A', 'C', 'G', 'T']): 
    #x_train data
    x = pd.read_csv(X_path)
    x = x['seq'].str.split('',expand = True)
    x = x.drop(columns =0)
    x = x.drop(columns = 102)
    x = x.values
    lst = []
    for item in seq: 
        lst.append(np.where(x == item,1,0))
    x_train = np.stack(lst,axis =2)

    #y_train 
    y = pd.read_csv(Y_path)
    y_train = y['Bound'].values

    return x_train, y_train.astype('float')

def test_set(test_path, seq = ['A', 'C', 'G', 'T']): 
    #x_test data
    x = pd.read_csv(test_path)
    x = x['seq'].str.split('',expand = True)
    x = x.drop(columns =0)
    x = x.drop(columns = 102)
    x = x.values
    lst = []
    for item in seq: 
        lst.append(np.where(x == item,1,0))
    x_test = np.stack(lst,axis =2)
    
    return x_test
