#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:34:59 2019

@author: lulumeng
"""
import numpy as np
    
# y_predict: array of 1s and 0s for the class prediction
# y: array of 1s and 0s for the true class label
# calculate the Accuracy, Precision, Recall, and F-1 measure
def calc_scores(y_predict, y):
    size = len(y)
    #true positive = true class is 1 and prediction is 1
    #true negative = true class is 0 and prediction is 0
    #false positive = true class is 0 and prediction is 1
    #false negative = true class is 1 and prediction is 0
    tp = 0
    tn = 0
    fn = 0
    for i in range(size):
        if (y[i]== 1) and (y_predict[i] == 1):
            tp += 1
        if (y[i]== 0) and (y_predict[i] == 0):
            tn += 1
        if (y[i]== 0) and (y_predict[i] == 1):
            fn += 1
    #in case tp+tn or tp+fn or recall+precision==0
    if tp==0:
        tp += 1
        print("true positive is 0, check the model")
    accuracy  = (tp+tn)/size
    precision = tp/(tp+tn+1)
    recall = tp/(tp+fn+1)
    f1 = 2*(recall*precision)/(recall+precision)
    return accuracy, precision, recall, f1


# split the data into 2 parts: traning set, validation set    
def fold(x, y, i, nfolds=10):
    
    n = len(x)
    
    n_left = round(n*(i/nfolds))
    n_right = round(n*((i+1)/nfolds))
    
    t1 = x[:n_left]
    t2 = x[n_left:n_right]
    t3 = x[n_right:]
    x_train = np.concatenate((t1,t3), axis=0)
    x_vali = t2
    
    t1 = y[:n_left]
    t2 = y[n_left:n_right]
    t3 = y[n_right:]
    y_train = np.concatenate((t1,t3), axis=0)
    y_vali = t2
    
    return x_train, y_train, x_vali, y_vali


#main function for cross validation
def cross_validation(x, y, classify_func, parameters):
    #initialize
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    nfolds = 10
    #calculate the scores of each folding and add them together
    for i in range(nfolds):
        x_train, y_train, x_test, y_test = fold(x, y, i, nfolds)
        y_predict = classify_func(x_train, y_train, x_test, parameters)
        acc_tmp, pre_tmp, rec_tmp, f1_tmp = calc_scores(y_predict, y_test)
        accuracy += acc_tmp/nfolds
        precision += pre_tmp/nfolds
        recall += rec_tmp/nfolds
        f1 += f1_tmp/nfolds
    #print
    print("accuracy is ",accuracy)
    print("precision is ",precision)
    print("recall is ",recall)
    print("f1 is ",f1)
