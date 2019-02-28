# @author: Cai Jie
# @Date:   2019/2/13 10:18

import pandas as pd
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def load_data():
    data = pd.read_csv("ex1data1.txt", sep=',', header=None)
    X = data[data.columns[0]].values
    y = data[data.columns[1]].values
    return X, y


def plot_data(x, y):
    plt.figure()
    plt.scatter(x, y, marker='x', color="r")
    plt.xlabel("population of cities")
    plt.ylabel("profit in $10000")
    plt.show()


def plot_linear_fit(X, y, theta):
    plt.figure()
    plt.scatter(X, y, marker='x', color="r")
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = theta[0] + theta[1] * x_vals
    plt.plot(x_vals, y_vals, 'b-')
    plt.xlabel("population of cities")
    plt.ylabel("profit in $10000")
    plt.show()


def cost_function(X, y, theta):
    m = len(y)
    hypo = X@theta
    J = (1/(2*m))*(hypo - y) @ (hypo - y)
    return J


def gradient_descend(X, y, alpha, theta, iterations):
    J_history = []
    m = len(y)
    for i in range(iterations):
        hypo = X @ theta
        J_gradient = X.T @ (hypo - y) * (1 / m)
        theta = theta - alpha*J_gradient
        J = cost_function(X, y, theta)
        J_history.append(J)
    return J_history, theta


def plot_J(J_history):
    plt.figure()
    plt.plot(np.arange(1, len(J_history)+1, 1), J_history)
    plt.xlabel("# of itertations")
    plt.ylabel("J")
    plt.show()


def plot_surf_J(X_plus, y, theta):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta_0 = np.linspace(-10, 10, 101)
    theta_1 = np.linspace(-1, 4, 101)

    Jvals =np.zeros((len(theta_0), len(theta_1)))
    for i in range(len(theta_0)):
        for j in range(len(theta_1)):
            Jvals[i, j] = cost_function(X_plus, y, np.array([theta_0[i], theta_1[j]]))
    theta_0, theta_1 = np.meshgrid(theta_0, theta_1, indexing='ij')

    surf = ax.plot_surface(theta_0, theta_1, Jvals, cmap=cm.viridis, antialiased=False)
    ax.set_xlabel("theta_0")
    ax.set_ylabel("theta_1")
    plt.show()

    plt.figure()
    plt.contour(theta_0, theta_1, Jvals, levels=np.logspace(-2, 3, 20), cmap=cm.viridis, antialiased=False)
    plt.plot(theta[0], theta[1], marker="x", color="r")
    plt.show()

def main():
    X, y = load_data()
    plot_data(X, y)
    X_plus = np.reshape(X, (len(X), 1))
    X_plus = np.insert(X_plus, 0, 1, axis=1)

    initial_theta = np.zeros(X_plus.shape[1])
    alpha = 0.01
    iterations = 1500

    # compute and display initial cost
    J = cost_function(X_plus, y, initial_theta)
    print("With theta = [0 ; 0] Cost computed = %.2f" % J)
    print("Expected cost value (approx) 32.07\n")

    # further testing of the cost function
    J = cost_function(X_plus, y, np.array([-1, 2]))
    print("With theta = [0 ; 0] Cost computed = %.2f" % J)
    print("Expected cost value (approx) 54.24\n")

    J_history, theta = gradient_descend(X_plus, y, alpha, initial_theta, iterations)
    plot_J(J_history)

    # print theta to screen
    print("Theta found by gradient descent: %f, %f" % (theta[0], theta[1]))
    print("Expected theta values (approx): -3.6303,1.1664\n")

    # Plot the linear fit
    plot_linear_fit(X, y, theta)

    # Plot surf J and contour
    plot_surf_J(X_plus, y, theta)









if __name__ == '__main__':
    main()