#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:21:42 2019

@author: SherryXiaoyingLi
"""
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import cross_vali
import data_preprocess

def fit(data, y, C):

    # method without using cvxopt

    # # contain tested parameters in a dict {||w|| : [w, b]}
    # # ||w|| represents value of the constraint function; [w, b] with the lowest ||w|| is desired
    # optimization_dict = {}
    #
    # # set w steps
    # transforms = [[1,1], [-1,1], [-1,-1], [1, -1]]
    # max_feature = max(max(entry) for entry in data)
    # min_feature = min(min(entry) for entry in data)
    # #start with 10% step to find the min, once finished, finer steps from there to find the min
    # steps = [max_feature * 0.1, max_feature * 0.01, max_feature * 0.001]
    #
    # # set b steps
    # b_range_multiple = 5
    # b_multiple = 5
    #
    # #try starting with weight = max feature value * 10
    # latest_weight = max_feature * 10
    #
    # p = len(data[0])
    # n = len(data)
    #
    # w = np.full((p, 1), latest_weight)
    # b = 0
    # for step in steps:
    #     # p number of features
    #     w = np.full((p, 1), latest_weight)
    #     optimized = False
    #     while not optimized:
    #
    #         for b in np.arange(-1 * max_feature * b_range_multiple,
    #                            max_feature * b_range_multiple, step * b_multiple ):
    #             for transformation in transforms:
    #                 w_t = w*transformation
    #                 found_option = True
    #
    #                 for i in range(n):
    #                     xi = data[i]
    #                     yi = y[i]
    #                     print(xi)
    #                     print(yi)
    #                     print(w_t)
    #                     print(b)
    #                     if not  (np.dot(w_t, xi) ) >= 1:
    #                         found_option = False
    #
    #                 if found_option:
    #                     optimization_dict[np.linalg.norm(w_t)] = [w_t, b]
    #         if w[0] < 0 :
    #             optimized = True
    #         else :
    #             w = w - step
    #
    #     norms = sorted([n for n in optimization_dict])
    #     choice = optimization_dict[norms[0]]
    #     w = choice[0]
    #     b = choice[1]
    #     latest_weight = choice[0][0] + step*2


    # method with cvxopt

    # C for additional constraints with non-linearly separable data
    m, n = data.shape
    y = y.reshape(-1, 1) * 1.
    x_dash = y * data
    H = np.dot(x_dash, x_dash.T) * 1.

    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))

    G = cvxopt_matrix(-np.eye(m))
    h = cvxopt_matrix(np.zeros(m))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    G_nonlinear = cvxopt_matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
    h_nonlinear = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))

    cvxopt_solvers.options['show_progress'] = False

    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

    sol_nonlinear = cvxopt_solvers.qp(P, q, G_nonlinear, h_nonlinear, A, b)
    alphas_nonlinear = np.array(sol_nonlinear['x'])

    # w, b based on the hints given on piazza
    w = ((y * alphas).T @ data).reshape(-1, 1)
    Support_vector = (alphas > 1e-4).flatten()
    b = y[Support_vector] - np.dot(data[Support_vector], w)

    w_nonlinear = ((y * alphas_nonlinear).T @ data).reshape(-1, 1)
    S_nonlinear = (alphas_nonlinear > 1e-4).flatten()
    b_nonlinear = y[S_nonlinear] - np.dot(data[S_nonlinear], w_nonlinear)


    # Display results
    print('Alphas = ', alphas[alphas > 1e-4])
    print('w = ', w.flatten())
    print('b = ', b[0])

    print('Alphas-nonlinear = ', alphas_nonlinear[alphas_nonlinear > 1e-4])
    print('w-nonlinear = ', w_nonlinear.flatten())
    print('b-nonlinear = ', b_nonlinear[0])

    return w.flatten(), b[0], w_nonlinear.flatten(), b_nonlinear[0]


def predict(data, w, b):
    # y = (x*w + b)
    class_result = np.sign(np.dot(data, w) + b)
    return class_result


def svm_classify(x, y, x_test, parameters):
    # change 0 in y -> -1
    y = np.where(y==0, -1, y)

    # fit with cvxopt function
    w, b, w_nonlinear, b_nonlinear = fit(x, y, parameters['C'])

    if parameters['type'] == 'linear-separable':
        prediction = predict(x_test, w, b)
        prediction = np.where(prediction == -1, 0, prediction)
        print(prediction)
        return prediction

    prediction_nonlinear = predict(x_test, w_nonlinear, b_nonlinear)
    prediction_nonlinear = np.where(prediction_nonlinear==-1, 0, prediction_nonlinear)
    print(prediction_nonlinear)
    return prediction_nonlinear

if __name__ == "__main__":
    x1, y1 = data_preprocess.data_read("project2_dataset1.txt")
    x1_sc = data_preprocess.data_norm(x1)

    x2, y2 = data_preprocess.data_read("project2_dataset2.txt")
    x2_sc = data_preprocess.data_norm(x2)
    #svm
    #for parameter 'type': try 'linearly-separable' or ''
    #if 'type' not 'linearly-separable', parameter 'C' needs to be set to a numeric value for
    # additional constraints on the optimization problem
    cross_vali.cross_validation(x1_sc, y1, svm_classify, {'type': 'lineraly-separable', 'C': 9})

