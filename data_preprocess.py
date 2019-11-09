#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:29:42 2019

@author: lulumeng
"""

import pandas as pd
from sklearn import preprocessing
from svm import svm
from neural import neural

def data_read(file_path):
    data_raw = pd.read_csv(file_path,sep=r'\t',header=None, engine='python')
    x_raw =  data_raw.iloc[:,:-1]
    y = data_raw.iloc[:,-1]
    # to do one-hot encoding for string column
    x = pd.get_dummies(x_raw)
    
    x = x.values
    y = y.values
    
    return x, y

def data_norm(data):
    #normoalize with min max scaler
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    return data_scaled


if __name__ == "__main__":
    #example of data loading
    x1,y1 = data_read("project2_dataset1.txt")
    x1_sc = data_norm(x1)

    x2,y2 = data_read("project2_dataset2.txt")
    x2_sc = data_norm(x2)

    #svm
    svm_pred, svm_pred_nonlinear = svm(x2_sc, y2)
    # neural_res = neural(data)
