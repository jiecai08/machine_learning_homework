# @author: Cai Jie
# @Date:   2019/2/22 16:11

import pandas as pd
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fmin_bfgs
from ex2_logistic_regression.ex2 import load_data,sigmoid


def plot_data(X, y):
    data_1 = X[y==1, :]
    data_0 = X[y==0, :]
    plt.figure()
    plt.scatter(data_1[:, 0], data_1[:, 1], marker='+', color='r', label='y = 1')
    plt.scatter(data_0[:, 0], data_0[:, 1], marker='o', color='b', label='y = 0')
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.legend()
    plt.show()


def map_feature(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    degree = 6
    x_maps = np.ones((X.shape[0], 1))
    for i in range(1, degree+1):
        for j in range(0, i+1):
            extra_feaure = x1**(i-j) * x2**j
            x_maps = np.hstack((x_maps, extra_feaure.reshape(X.shape[0], 1)))
    return x_maps


def cost_reg(theta, X, y, lambd):
    m = len(X)
    z = X @ theta
    cost = (-1/m) * (y.T @ np.log(sigmoid(z))+(1-y).T @ np.log(1-sigmoid(z)))
    regular = (lambd/(2*m))*theta @ theta
    return cost+regular


def grad_reg(theta, X, y, lambd):
    m = len(X)
    z = X @ theta
    J_grad = (1/m)*X.T @ (sigmoid(z)-y)+(lambd/m)*theta
    J_grad[0] = J_grad[0] - (lambd/m)*theta[0]          # WARNING: theta0 do not need add regularization items
    return J_grad


def plot_decison_boundary(X, y, theta):
    data_1 = X[y == 1, :]
    data_0 = X[y == 0, :]
    plt.figure()
    plt.scatter(data_1[:, 0], data_1[:, 1], marker='+', color='r', label='y = 1')
    plt.scatter(data_0[:, 0], data_0[:, 1], marker='o', color='b', label='y = 0')
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")

    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = map_feature(np.array([[u[i], v[j]]])) @ theta
    u, v = np.meshgrid(u, v, indexing='ij')
    dec_bound = plt.contour(u, v, z, levels=0, antialiased=False)
    dec_bound.collections[0].set_label("decision boundary")
    plt.legend()
    plt.show()


def predict(theta, X, y):
    x_maps = map_feature(X)
    predict = sigmoid(x_maps @ theta)
    predict[predict>=0.5] = 1
    predict[predict<0.5] = 0
    accuracy = np.mean(predict == y)
    return accuracy





def main():
    # load data
    filename = "ex2data2.txt"
    X, y = load_data(filename)

    # visualize data
    plot_data(X, y)

    # map feature
    x_maps = map_feature(X)

    # initial theta and lambda
    theta = np.zeros(x_maps.shape[1])
    lambdas = 1

    # verify cost_reg at initial theta
    J = cost_reg(theta, x_maps, y, lambdas)
    J_grad = grad_reg(theta, x_maps, y, lambdas)

    print("cost at initial theta is %f" % J)
    print("Expected cost (approx): 0.693")
    print("Gradient at initial theta (zeros) - first five values only")
    print(J_grad[0:5])
    print("Expected gradients (approx) - first five values only")
    print("0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n")

    opt_theta = fmin_bfgs(cost_reg, theta, fprime=grad_reg, args=(x_maps, y, lambdas), maxiter=400)
    plot_decison_boundary(X, y, opt_theta)

    # Compute accuracy on our training set
    accuracy = predict(opt_theta, X, y)
    print(accuracy)
    print("Train accuracy is {0:.1%}".format(accuracy))
    print("Expected accuracy (with lambda = 1): 83.1 (approx)")







if __name__ == '__main__':
    main()