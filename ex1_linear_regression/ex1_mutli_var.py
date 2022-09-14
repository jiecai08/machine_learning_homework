# @author: Cai Jie
# @Date:   2019/2/13 18:14

from ex1_linear_regression.ex1 import gradient_descend, plot_J
import pandas as pd
import numpy as np
from scipy.linalg import pinv


def load_data():
    data = pd.read_csv("ex1data2.txt", sep=',', header=None)
    X = data.iloc[:, 0:len(data.columns)-1].values
    y = data[data.columns[-1]].values
    return X, y


def feature_normalization(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu)/sigma
    return X_norm, mu, sigma



def main():
    X, y = load_data()
    # gradient need feature scaling
    X_norm, mu, sigma = feature_normalization(X)
    X_plus = np.insert(X_norm, 0, 1, axis=1)

    init_theta = np.zeros(X_plus.shape[1])
    alpha = 0.01
    iterations = 400
    J_history, theta = gradient_descend(X_plus, y, alpha, init_theta, iterations)

    # plot the convergence graph
    plot_J(J_history)
    input("Press Enter to continue...")

    # display gradient descend result
    print("Theta computed from Gradient descend is:")
    print(theta)
    print("\n")

    # Estimate the price of 1650 sq-ft, 3 br house
    x = (np.array([1650, 3]) - mu)/sigma
    price = theta @ np.insert(x, 0, 1)
    print("The predicted price of 1650 sq-ft, 3 br house is: %f\n" % price)


    # Norm Equation
    X = np.insert(X, 0, 1, axis=1)
    theta = pinv(X.T @ X) @ X.T @ y

    print("Theta computed by normal equation is")
    print(theta)
    print("\n")

    # Estimate the price of 1650 sq-ft, 3 br house
    price = theta @ np.array([1, 1650, 3])
    print("The predicted price of 1650 sq-ft, 3 br house is: %f\n" % price)














if __name__ == '__main__':
    main()