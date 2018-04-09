#!/anaconda/envs/tensorflow/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:30:48 2018

@author: ZHANG Pingcheng
"""

from multiprocessing import Queue, Process, cpu_count
from collections import Iterable
from time import time
import numpy as np


def flatten(nested):
    return list(filter(lambda _: _,
     (lambda _: ((yield from flatten(
        e)) if isinstance(e,
         Iterable) else (yield round(e, 6)) for e in _))(nested)))


onezero = np.vectorize(lambda x: 0 if x < 0.5 else 1)
safelog = np.vectorize(lambda x: x if x != 0 else 0.00000000001)
sigmoid = lambda z: 1 / (1 + np.exp(-1 * z))
ttsplit = lambda par, y, X: (y[:par], X[:par], y[par:], X[par:])
average = lambda ls: sum([s[1] for s in ls]) / len(ls)
converged = lambda temp, theta, tol: abs(np.sum(temp - theta)) <= tol


# Helper functions:
def traincsv2matrix(file: str) -> (np.matrix, np.matrix):
    """
    Retrieve training data from csv file.
    Params:
        file: A string indicating a file address
    Return:
        y and X
    """
    data = np.matrix(np.genfromtxt(file, delimiter=','))
    return data[:, -1], data[:, :-1]


def predict2csv(y: np.matrix) -> None:
    """
    Write the predicted y into csv file.
    Params:
            y: Column vector of y values
    """
    np.savetxt("predicted_y.csv", y)


def cv_divide(y: np.matrix, X: np.matrix, n: int, index: int):
    """
    Helper function to create training sets and testing set for n-fold validation.
    No extra spaces are alloacated, just indexing.
    Jesus it's so ugly.
    Params:
            y: Column vector of y values
            X: Matrix of x values
            n: Number of folds
        index: The index of one batch out of n batches
    Return:
        A batch in the form of train:test
    """
    m = y.shape[0]
    batch_size = m // n
    last_batch = m % n + batch_size
    if index not in range(n):
        raise IndexError("Batch number out of range.")
    if index == 0:
        y_test = y[:(batch_size * (index + 1))]
        X_test = X[:(batch_size * (index + 1))]
        y_train = y[(batch_size * (index + 1)):]
        X_train = X[(batch_size * (index + 1)):]
    elif index == (n - 1):
        y_test = y[-last_batch:]
        X_test = X[-last_batch:]
        y_train = y[:m - last_batch]
        X_train = X[:m - last_batch]
    else:
        y_test = y[(batch_size * index):(batch_size * (index + 1))]
        X_test = X[(batch_size * index):(batch_size * (index + 1))]
        y_train = np.vstack((y[:batch_size * index],
                             y[batch_size * (index + 1):]))
        X_train = np.vstack((X[:batch_size * index],
                             X[batch_size * (index + 1):]))
    return y_train, X_train, y_test, X_test


def batch_devide(y: np.matrix, X: np.matrix, n: int, index: int) -> np.matrix:
    """
    Helper function to do parallelized calculation of gradient descent.
    Partition the training set only.
    Params:
            y: Column vector of y values
            X: Matrix of x values
            n: Number of parallel processes, normally the number of CPU cores
        index: The index of one batch out of n batches
    """
    m = y.shape[0]
    batch_size = m // n
    last_batch = m % n + batch_size
    if index not in range(n):
        raise IndexError("Batch number out of range.")
    if index == 0:
        mini_y = y[:(batch_size * (index + 1))]
        mini_X = X[:(batch_size * (index + 1))]
    elif index == (n - 1):
        mini_y = y[-last_batch:]
        mini_X = X[-last_batch:]
    else:
        mini_y = y[(batch_size * index):(batch_size * (index + 1))]
        mini_X = X[(batch_size * index):(batch_size * (index + 1))]
    return mini_y, mini_X


# Core parts: cost, gradient, descent, error, predict
def cost(y: np.matrix, X: np.matrix, theta: np.matrix, lamda: float) -> float:
    """
    Calculates cost function.
    It's not safe to use the defaut np.log to calculate cost function.
    Params:
            y: Column vector of y values
            X: Matrix of x values
        theta: A column vector of coefficients of logistic regression model
    Return:
        A float number cost
    """
    m = y.shape[0]
    hypo = sigmoid(X @ theta)
    # Note that we need a number only. So we retrived value using [0, 0]
    # The following line neet to add regularization part to avoid overfitting
    c = (-1.0 / m * (y.T @ np.log(safelog(hypo)) +
                     (1.0 - y).T @ np.log(safelog(1.0 - hypo))))[0, 0]
    # L2 regularization:
    return c + lamda / (2 * m) * np.sum(np.square(
        theta[1:]))  # Bug can show up if len(theta) < 2


def gradient(y: np.matrix, X: np.matrix, theta: np.matrix,
             lamda: float) -> np.matrix:
    """
    Calculates gradient. Gradient measures how weights affects cost.
    Params:
            y: Column vector of y values
            X: Matrix of x values
        theta: A column vector of coefficients of logistic regression model
    Return:
        A column vector gradient
    """
    m = y.shape[0]
    hypo = sigmoid(X @ theta)
    L = np.matrix(np.identity(X.shape[1]))
    L[0, 0] = 0
    # Here though, unlike line 169, we must preserve the structure of the gradient.
    return 1.0 / m * (X.T @ (hypo - y) + lamda * L @ theta)


# Gradient descent algorithm in multiprocessing: parallel gradient calculation.
def para_gradient(y: np.matrix, X: np.matrix, theta: np.matrix) -> float:
    """
    Calculate gradient in parallel.
    
    Using the Map-Reduce approach, 'para' is abbreviation for parallelized
    Params:
            y: Column vector of y values
            X: Matrix of x values
        theta: A column vector of coefficients of logistic regression model
    Return:
        A column vector gradient

    WARNING: Do not use this version unless dataset is very large!
    """
    core_num = cpu_count()
    m = y.shape[0]
    q = Queue()
    tasks = []
    gradients = []
    for c in range(core_num):
        para_y, para_X = batch_devide(y, X, core_num, c)
        p = Process(target=cthgrad, args=(para_y, para_X, theta, q))
        tasks.append(p)
        p.start()
    for b in tasks:
        b.join()
    while not q.empty():
        gradients.append(q.get())
    return 1.0 / m * sum(gradients)


def cthgrad(y: np.matrix, X: np.matrix, theta: np.matrix, queue: Queue):
    """
    Calculates gradient of the devided sub-batch of dataset, then to be merged in
    its outer caller function.
    Params:
            i: The i-th subbatch
            n: Number of batches 
            y: Column vector of y values
            X: Matrix of x values
        queue: Multiprocessing queue to store calculation result
        alpha: Learning rate
          tol: Tolerance of minimum update
    """
    hypo = sigmoid(X @ theta)
    queue.put(X.T @ (hypo - y))


def descent(y: np.matrix,
            X: np.matrix,
            alpha: float,
            tol: float,
            lamda: float,
            maxiter=np.inf) -> np.matrix:
    """
    Vectorized implementation of radient descent algorithm.
    Stop condition is when the theta barely changes.
    Params:
            y: Column vector of y values
            X: Matrix of x values
        alpha: Learning rate
          tol: Tolerance of minimum update
      maxiter: Maximum number of iteration, defaut to possitive infinity.
    Return:
        theta: A column vector in the form of np.matrix
    """
    i = 1
    temp = np.zeros((X.shape[1], 1))
    theta = np.ones((X.shape[1], 1))
    loss = cost(y, X, theta, lamda)
    while not converged(temp, theta, tol) and i < maxiter:
        if i // 100 == 0:
            print("Iteration {0} | cost: {1: .6f}".format(i, loss))
        temp = theta
        theta = theta - alpha * gradient(y, X, theta, lamda)
        # theta = theta - alpha * para_gradient(y, X, theta)
        loss = cost(y, X, theta, lamda)
        i += 1
    return theta


def error(theta: np.matrix, y_test: np.matrix, X_test: np.matrix) -> float:
    """
    Prediction accuracy.
    Params:
        theta: Column vector of coefficients
       y_test: testing set ys
       X_test: testing set X
    Return:
        A float number gives the error of prediction
    """
    y_predict = onezero(X_test @ theta)
    return abs(np.sum(y_predict - y_test)) / y_test.shape[0]


def predict(theta: np.matrix, X_input: np.matrix) -> np.matrix:
    """
    Given trained theta and input X, predict y.
    Applicable only when the dataset is small.
    Params:
        theta: Column vector of coefficients
      X_input: Matrix of input data to be multiplied with theta to produce predicted ys
    Return:
            y: A column vector of predicted values
    """
    return onezero(X_input @ theta)


# Cross Validation:
def nfold(n: int, y: np.matrix, X: np.matrix, alpha: float, tol: float,
          lamda: float) -> (float, (np.matrix, float)):
    """
    N-fold cross validation.
    Params:
            n: Number of batches
            y: Column vector of y values
            X: Matrix of x values
        queue: Multiprocessing queue to store calculation result
        alpha: Learning rate
          tol: Tolerance of minimum update
    Return:
        A tuple (error, theta)
    """
    errors = []
    for i in range(n):
        print("N-Fold Validation -> {}:".format(i))
        y_train, X_train, y_test, X_test = cv_divide(y, X, n, i)
        theta = descent(y_train, X_train, alpha, tol, lamda)
        e = error(theta, y_test, X_test)
        errors.append((theta, e))
        print("{0} -> theta: {1} | error: {2:.6f}".format(i, theta, e))
    return average(errors), min(errors, key=lambda x: x[1])


# Parallel version of n-fold:
def multifold(n: int, y: np.matrix, X: np.matrix, alpha: float, tol: float,
              lamda: float) -> (float, (np.matrix, float)):
    """
    Multiprocessing N-fold cross validation. It calls function kthfold in parallel.
    Params:
            n: Number of batches
            y: Column vector of y values
            X: Matrix of x values
        queue: Multiprocessing queue to store calculation result
        alpha: Learning rate
          tol: Tolerance of minimum update
    Return:
        A tuple (error, theta)
    """
    q = Queue()
    trains = []
    errors = []
    for k in range(n):
        print("N-Fold Validation -> {}:".format(k))
        p = Process(target=kthfold, args=(k, n, y, X, q, alpha, tol, lamda))
        trains.append(p)
        p.start()
    for t in trains:
        t.join()
    while not q.empty():
        errors.append(q.get())
    return average(errors), min(errors, key=lambda x: x[1])


def kthfold(i: int, n: int, y: np.matrix, X: np.matrix, queue: Queue,
            alpha: float, tol: float, lamda: float):
    """
    The kth partition helper function for N-fold cross validation.
    Params:
            i: The i-th subbatch
            n: Number of batches
            y: Column vector of y values
            X: Matrix of x values
        queue: Multiprocessing queue to store calculation result
        alpha: Learning rate
          tol: Tolerance of minimum update
    """
    y_train, X_train, y_test, X_test = cv_divide(y, X, n, i)
    theta = descent(y_train, X_train, alpha, tol, lamda)
    e = error(theta, y_test, X_test)
    print("{0} -> theta: {1} | error: {2:.6f}".format(i, flatten(
        theta.tolist()), e))
    queue.put((theta, e))


# Callable kick-starting functions:
def simple_split(y: np.matrix, X: np.matrix, p: float, alpha: float,
                 tol: float, lamda: float) -> (np.matrix, float):
    """
    Simple training and testing the model using 2-8 partition.
    Params:
            y: Column vector of y values
            X: Matrix of x values
            p: Proportion of dataset serving as training set
        alpha: Learning rate
          tol: Tolerance of minimum update
    """
    start = time()
    y_train, X_train, y_test, X_test = ttsplit(int(y.shape[0] * p), y, X)
    theta = descent(y_train, X_train, alpha, tol, lamda)
    e = error(theta, y_test, X_test)
    end = time()
    print("Theta    : {}".format(flatten(theta.tolist())))
    print("Accuracy : {0:.6f}".format(1 - e))
    print("Total elapsed time: {}".format(end - start))
    # Just to stay compatible with other two fucntion returning format:
    return e, [theta]


def nfold_train(y: np.matrix, X: np.matrix, n: int, alpha: float, tol: float,
                lamda: float) -> (float, (np.matrix, float)):
    """
    Sequencial training process of n-fold cross validation.
    Params:
            y: Column vector of y values
            X: Matrix of x values
            n: Number of batches
        queue: Multiprocessing queue to store calculation result
        alpha: Learning rate
          tol: Tolerance of minimum update
    """
    start = time()
    aveg_err, best_theta = nfold(5, y, X, alpha, tol, lamda)
    print("Average accuracy: {:.6f}".format(1 - aveg_err))
    print("Highest accuracy: {:.6f}".format(1 - best_theta[1]))
    print("Theata: {}".format(flatten(best_theta[0].tolist())))
    end = time()
    print("Total elapsed time: {}".format(end - start))
    return aveg_err, best_theta


def multifold_train(y: np.matrix, X: np.matrix, n: int, alpha: float,
                    tol: float, lamda: float) -> (float, (np.matrix, float)):
    """
    Parallelized training process of n-fold cross validation.
    Params:
            y: Column vector of y values
            X: Matrix of x values
            n: Number of batches
        queue: Multiprocessing queue to store calculation result
        alpha: Learning rate
          tol: Tolerance of minimum update
    """
    start = time()
    aveg_err, best_theta = multifold(n, y, X, alpha, tol,
                                     lamda)  # 7 is the number
    print("Average accuracy: {:.6f}".format(1 - aveg_err))
    print("Highest accuracy: {:.6f}".format(1 - best_theta[1]))
    print("Theta: {}".format(flatten(best_theta[0].tolist())))
    end = time()
    print("Total elapsed time: {}".format(end - start))
    return aveg_err, best_theta


if __name__ == "__main__":

    # Read data:
    y, X = traincsv2matrix("diabetes_dataset.csv")

    # Simple training and testing the model:
    """avge, best_theta = simple_split(y, X, 0.8, 0.00001, 0.0001, 1)"""

    # Using N-fold validation strategy:
    """avge, best_theta = nfold_train(y, X, 7, 0.0001, 0.0001, 1)"""

    # Multiprocessing N-fold
    avge, best_theta = multifold_train(y, X, 7, 0.0001, 0.0001, 50)

    # Predict y on non-labeled dataset:
    theta = best_theta[0]
    e = error(theta, y, X)
    print("Accuracy using best theta: {:.6f}".format(1 - e))
    predict2csv(
        predict(theta,
                np.matrix(np.genfromtxt("test_samples.csv", delimiter=','))))
