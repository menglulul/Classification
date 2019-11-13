#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:35:25 2019

@author: lulumeng
"""
import operator
import numpy as np
import cross_vali
import data_preprocess

def knn_classify(x_train, y_train, x_test, k):
    n_test = len(x_test)
    n_train = len(x_train)
    y_predict = []
    for i in range(n_test):
        distances = []
        for j in range(n_train):
            distances.append((cal_distance(x_train[j],x_test[i]), y_train[j]))
        distances.sort(key=operator.itemgetter(0))
        neighbor_cnt = {}
        for j in range(k):
            neighbor = distances[j][1]
            if neighbor in neighbor_cnt:
                neighbor_cnt[neighbor] += 1
            else:
                neighbor_cnt[neighbor] = 1
        sorted_cnt = sorted(neighbor_cnt.items(), key=operator.itemgetter(1), reverse=True)
        y_predict.append(sorted_cnt[0][0])
    return y_predict

def cal_distance(x, y):
    return np.linalg.norm(x-y)

if __name__ == "__main__":
        # read data
    x1,y1 = data_preprocess.data_read("project2_dataset1.txt")
    x1_sc = data_preprocess.data_norm(x1)

    x2,y2 = data_preprocess.data_read("project2_dataset2.txt")
    x2_sc = data_preprocess.data_norm(x2)
    
    for k in range(1,15):
        print("k=",k)
        print("dataset1")
        cross_vali.cross_validation(x1, y1, knn_classify, k)
        cross_vali.cross_validation(x1_sc, y1, knn_classify, k)
        print("dataset2")
        cross_vali.cross_validation(x2, y2, knn_classify, k)
        cross_vali.cross_validation(x2_sc, y2, knn_classify, k)