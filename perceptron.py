#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 18:22:05 2018

@author: spencer

Perceptron learing algorithm in 2d
As a presentation of the idea
This algorithm will be generalized in to suit more realistic scenarios.
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


# definition of the perceptron learning algorithm
def perceptron(Input, label, w, miss):
    """
    Input: list of input value, [x1, x2, x3,...,xn]
    label: + or -
    w: initial weights for training
    """
    # print('w->', w)
    global track
    assert len(Input) == len(label)
    for i in range(len(Input)):
        x = [1]
        x.extend(Input[i])
        # print('np.dot(w, x):', np.dot(w, x), 'label[i]: ', label[i])
        if np.dot(w, x) * label[i] < 0:
            miss[i] = 1
            w = list(np.add(w, np.multiply(x, label[i]))) # add up to vector
            track += 1
        else:
            miss[i] = 0
    return w, miss


def train(p, Input, label, w):
    miss = [1] * N
    print('miss before iter:', miss)
    print('inital weight:', w)
    iteration = 1
    while sum(miss) >= N * 0.01:
        print('interation {}'.format(iteration))
        new_w, miss = p(Input, label, w, miss)
        print('new_w -> ', new_w)
        w = new_w
        iteration += 1
        print('miss at iteration {}'.format(iteration), miss)
    miss = [1] * N
    return w


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
    for i in gen_pts(N):
        # print(i)
        xs.append([i[0], i[1]])
        ys.append(i[2])
        plt.plot(i[0], i[1], *['g+' if i[2]>0 else 'b_'])
    
    plt.plot([-1, 0], [0, -1], 'r--', linewidth=3)
    
    w_hypoth = train(perceptron,
                     xs,
                     ys,
                     [rd.randint(-10,10),
                      rd.randint(-10,10),
                      rd.randint(-10,10)
                      ]
                     )

    print('w_truth -> ', w_truth)
    print('w_hypoth -> ', w_hypoth)
    print('track(or the actual iteration): ', track)
    
    # plot the true line
    plt.plot([0, -w_hypoth[0]/w_hypoth[2]], [-w_hypoth[0]/w_hypoth[1], 0],
             'y-', linewidth=2, label="hypothesis")
    
    plt.show()