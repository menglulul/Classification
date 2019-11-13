#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:49:40 2019

@author: lulumeng
"""
import numpy as np
import cross_vali
import data_preprocess
import math

#return mean(2*k length array), variance(2*k length array)
def cal_mean_var(x_train, y_train):
    k = len(x_train[0])
    n = len(x_train)
    
    mean = np.zeros((2,k))
    var = np.zeros((2,k))
    cnt0 = 0
    cnt1 = 0
    
    for i in range(n):
        if y_train[i] == 0:
            mean[0] += x_train[i]
            cnt0 += 1
        else:
            mean[1] += x_train[i]
            cnt1 += 1
    mean[0] /= cnt0
    mean[1] /= cnt1
    
    for i in range(n):
        if y_train[i] == 0:
            for j in range(k):
                var[0][j] += (x_train[i][j]-mean[0][j])**2
        else:
            for j in range(k):
                var[1][j] += (x_train[i][j]-mean[1][j])**2
    var[0] /= cnt0
    var[1] /= cnt1
    return mean, var


#return 2*1 array
def cal_pre_probability(y_train):
    pre_prob = np.zeros(2)
    pre_prob[0] = 1 - np.sum(y_train)/len(y_train)
    pre_prob[1] = np.sum(y_train)/len(y_train)
    return pre_prob

def cal_cond_prob(x, mean, var):
	exponent = math.exp(-((x-mean)**2 / (2 * var )))
	return (1 / (math.sqrt(2 * math.pi) * var)) * exponent

def cal_probability(x, mean, var, preprob):
    prob = np.ones(2)
    k = len(x)
    for i in range(k):
        for j in range(2):
            prob[j] *= cal_cond_prob(x[i],mean[j][i],var[j][i])
    for j in range(2):
        prob[j] *= preprob[j]
        
    return prob



def nb_classify(x_train, y_train, x_test, para=None):
    mean, var = cal_mean_var(x_train, y_train)
    preprob = cal_pre_probability(y_train)
    y_predict = []
    for test_point in x_test:
        prob = cal_probability(test_point, mean, var, preprob)
        y_predict.append(prob.argmax())
    return y_predict
    

if __name__ == "__main__":
        # read data
    x1,y1 = data_preprocess.data_read("project3_dataset1.txt")
    x1 = data_preprocess.data_norm(x1)
    cross_vali.cross_validation(x1, y1, nb_classify, 0)
    
    x2,y2 = data_preprocess.data_read("project3_dataset2.txt")
    x2 = data_preprocess.data_norm(x2)
    cross_vali.cross_validation(x2, y2, nb_classify, 0)