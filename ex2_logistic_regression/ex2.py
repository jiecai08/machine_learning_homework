# @author: Cai Jie
# @Date:   2019/2/22 10:54

import pandas as pd
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fmin_bfgs


def load_data(filename):
    df = pd.read_csv(filename, sep=',', header=None)
    X = df.iloc[:, 0: len(df.columns)-1].values
    y = df[df.columns[-1]].values
    return X, y


def plot_data(X, y):
    data_1 = X[y==1, :]
    data_0 = X[y==0, :]
    plt.figure()
    plt.scatter(data_1[:, 0], data_1[:, 1], marker='+', color='r', label='admitted')
    plt.scatter(data_0[:, 0], data_0[:, 1], marker='o', color='b', label='not admitted')
    plt.xlabel("exam1 score")
    plt.ylabel("exam2 score")
    plt.legend()
    plt.show()


def sigmoid(z):
    hypo = 1.0/(np.exp(-z)+1.0)
    return hypo


def cost(theta, X, y):
    m = len(y)
    epsilon = 1e-15
    hypo = sigmoid(X@theta)
    J = (-1/m) * (y @ np.log(hypo+epsilon) + (1-y) @ np.log(1-hypo+epsilon))
    return J


def grad(theta, X, y):
    m = len(y)
    hypo = sigmoid(X @ theta)
    J_gradient = (1 / m) * (X.T @ (hypo - y))
    return J_gradient


def plot_decision_boundary(X, y, theta):
    data_1 = X[y == 1, :]
    data_0 = X[y == 0, :]
    plt.figure()

    plt.scatter(data_1[:, 0], data_1[:, 1], marker='+', color='r', label='admitted')
    plt.scatter(data_0[:, 0], data_0[:, 1], marker='o', color='b', label='not admitted')
    plt.xlabel("exam1 score")
    plt.ylabel("exam2 score")
    plt.legend()
    ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = (theta[0]+theta[1]*x_vals)/-theta[2]
    plt.plot(x_vals, y_vals, "--")
    plt.show()




def main():
    # Load Data with pandas
    filename = "ex2data1.txt"
    X, y = load_data(filename)
    X_plus = np.insert(X, 0, 1, axis=1)

    # visualizing the data
    plot_data(X, y)
    input("Press Enter to continue...")

    # initialize thetas
    init_thetas = np.zeros(X_plus.shape[1])
    J = cost(init_thetas, X_plus, y)
    J_gradient = grad(init_thetas, X_plus, y)

    print("cost at initial theta is %f" % J)
    print("Expected cost is: 0.693")
    print("Gradient at initial theta is:")
    print(J_gradient)
    print("Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n")
    input("Press Enter to continue...")

    # valid cost and gradient with other initial theta
    test_theta = np.array([-24, 0.2, 0.2])
    J = cost(test_theta, X_plus, y)
    J_gradient = grad(test_theta, X_plus, y)
    print("cost at initial theta is %f" % J)
    print("Expected cost is: 0.218")
    print("Gradient at initial theta is:")
    print(J_gradient)
    print("Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n")
    input("Press Enter to continue...")

    # optimization using optimize.fmin_bfgs
    np.seterr(all='raise')
    opt_theta = fmin_bfgs(cost, init_thetas, fprime=grad, args=(X_plus, y))
    print("Expected cost (approx): 0.203")
    print("optimal theta is")
    print(opt_theta)
    print("Expected theta (approx):\n-25.161\n 0.206\n 0.201\n")

    # plot decision boundary
    plot_decision_boundary(X, y, opt_theta)

    # predict and get accuracy
    prob = sigmoid(np.array([1, 45, 85])@opt_theta)
    print("For a student with scores 45 and 85, we predict an admission probability is %f" % prob)
    print("Expected value: 0.775 +/- 0.002\n")

    # calculate accuracy with training sets
    p = sigmoid(X_plus @ opt_theta)
    p[p>=0.5] = 1
    p[p<0.5] = 0
    accuracy = np.mean(p == y) * 100
    print("Train accuracy :%f percent" % accuracy)
    print("Expected accuracy (approx): 89.0\n")





if __name__ == '__main__':
    main()