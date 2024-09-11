# Author: edit by Sara Schmidgall
# Date: 02.09.2024
# Project: DMML
# Acknowledgements: 
#

import torch
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal


def mvn_basis(
    features: torch.Tensor,
    mu: torch.Tensor,
    var: float
) -> torch.Tensor:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * var: All normal distributions are isotropic with sigma^2*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    solution = []
    for j in range(features.shape[0]):
        saver = []
        for i in range(mu.shape[0]):
            solution2 = multivariate_normal.pdf(features[j], mu[i], var)
            saver.append(solution2)
        solution.append(saver)
    solution = torch.tensor(solution)
    return solution

def _plot_mvn(fi):
    plt.clf
    for i in range(fi.shape[1]):
        plt.plot(fi[:,i], label='basic fcn'+str(i))
    plt.legend(loc='upper left')
    plt.show()


def max_likelihood_linreg(
    fi: torch.Tensor,
    targets: torch.Tensor,
    lamda: float
) -> torch.Tensor:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    targets = targets.to(torch.float64) 
    identity = torch.eye(fi.shape[1])
    multiplicand_1 = torch.inverse((lamda * identity) + torch.matmul(torch.transpose(fi, 0, 1) , fi))
    multiplicand_2 = torch.matmul(multiplicand_1, torch.transpose(fi, 0, 1))
    multiplicand_2 = torch.matmul(multiplicand_2, targets)
    multiplicand_2 = multiplicand_2.to(torch.float32)
    return multiplicand_2


def linear_model(
    features: torch.Tensor,
    mu: torch.Tensor,
    var: float,
    w: torch.Tensor
) -> torch.Tensor:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * var: All normal distributions are isotropic with sigma^2*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    w = w.to(torch.float64) 
    fi = mvn_basis(features, mu, var)
    w = torch.t(w)
    fi = fi.transpose(0,1)
    y = torch.matmul(w, fi)
    y = y.to(torch.float32)
    return y


if __name__ == "__main__":
    # Section 1
    X, t = load_regression_iris()
    N, D = X.shape
    M, var = 10, 10
    mu = torch.zeros((M, D))
    for i in range(D):
        mmin = torch.min(X[:, i])
        mmax = torch.max(X[:, i])
        mu[:, i] = torch.linspace(mmin, mmax, M)

    fi = mvn_basis(X, mu, var)

    # Section 2
    #_plot_mvn(fi)

    # Section 3
    lamda = 0.001
    wml = max_likelihood_linreg(fi, t, lamda)

    # Section 4
    prediction = linear_model(X, mu, var, wml)

    # Section 5
    plt.clf()
    plt.plot(t, label='target')
    plt.plot(prediction, label='prediction')
    plt.legend(loc='upper left')
    plt.show()

    difference = torch.abs(t - prediction)
    plt.clf()
    plt.plot(difference)
    plt.title('absolut diff = |target - prediction|')
    plt.show()

    # mean-square-error
    # 1/n * sum((t-pred)^2)
    sum = 0
    for i in range(t.shape[0]):
        sum = sum + torch.square(t[i] - prediction[i])
    meanSquare = 1/t.shape[0] * sum
    print(meanSquare)

    # 1/2 * (t-pred)^2
    mse = 1/2  * torch.square(t - prediction)
    plt.clf()
    plt.plot(mse)
    plt.title('mean square error')
    plt.show()

    
    pass
