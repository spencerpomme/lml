#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:07:51 2018

@author: spencer

Use the 'one-step' linear regression method to solve w. Actually, it used linear
algebra to solve linear equations.
"""
import numpy as np
import matplotlib.pyplot as plt
import random as rd


# Using the randomly generated parameter to define the real target function
def target(x, y):
    # print(np.dot(w_truth, [1, x, y]))
    if np.dot(w_truth, [1, x, y]) > 0:
        return 1
    else:
        return -1
    # eturn 1 if np.dot(w_truth, np.transpose([1,x,y])) > 0 else -1


# Generate n random points according to the target function
def gen_pts(n: int):
    while n > 0:
        n -= 1
        x = rd.uniform(-1, 1)
        y = rd.uniform(-1, 1)
        yield [x, y, target(x, y)]


def linear_solver(X: np.matrix, y: np.matrix) -> list:
    """
    Input: list of x        
    label: list of y
    """
    hypoth = (np.transpose(X) * X).I * np.transpose(X) * np.transpose(y)
    return [hypoth.tolist()[0][0], hypoth.tolist()[1][0], hypoth.tolist()[2][0]]


def error_in(w_hypoth: list, xs: list, ys: list) -> float:
    """
    Add a error measure function (i.e. cost/loss/error function)
    Measures the in-sample error.
    """
    error = 0
    total = len(ys)
    for i in range(total):
        if (1 if np.dot(w_hypoth, xs[i]) > 0 else -1) != ys[i]:
            error += 1
    return error / total


# No need for a train function here because linear_solver is one-step

if __name__ == "__main__":
    # maigc number don't delete
    track = 0
    # Number of points
    N = 100
    # Set line parameters: x + y + 1= 0
    A = 1
    B = 1
    C = 1
    w_truth = [C, A, B]

    # The sample set {xs -> ys}
    xs = []
    ys = []
    # control points of line x+y+1=0:
    p1 = [-1, 0]
    p2 = [0, -1]
    # plot the points
    for e in range(100):
        errors = []
        for i in gen_pts(N):
            # print(i)
            xs.append([1, i[0], i[1]])
            ys.append(i[2])
            plt.plot(i[0], i[1], *['g+' if i[2] > 0 else 'b_'])

        plt.plot([-1, 0], [0, -1], 'r--', linewidth=3)
        w_hypoth = linear_solver(np.matrix(xs), np.matrix(ys))

        # print("f ->", w_truth)
        # print("g ->", w_hypoth)
        errors.append(error_in(w_hypoth, xs, ys))
        print("in-sample error -> ", error_in(w_hypoth, xs, ys))
        plt.plot(
            [-w_hypoth[0] / w_hypoth[1], 0], [0, -w_hypoth[0] / w_hypoth[2]],
            'y-',
            linewidth=2,
            label="hypothesis")

        plt.show()
    print(sum(errors) / 100)
    # print(type(w_hypoth.tolist()[0][0]))
    # print(w_hypoth.tolist()[1][0])
