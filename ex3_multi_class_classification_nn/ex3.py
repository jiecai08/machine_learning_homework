# @author: Cai Jie
# @Date:   2019/2/28 15:56

import pandas as pd
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fmin_bfgs
import scipy.io as sio
import math


def load_data():
    data = sio.loadmat("ex3data1.mat")
    X = data['X']
    y = data['y']
    return X, y


def main():
    # load data
    X, y = load_data()
    print(X[0])

    # visualize







if __name__ == '__main__':
    main()