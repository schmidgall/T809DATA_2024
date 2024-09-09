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
    sigma: float
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
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    solution = []
    for j in range(features.shape[0]):
        saver = []
        for i in range(mu.shape[0]):
            solution2 = multivariate_normal.pdf(features[j], mu[i], sigma)
            saver.append(solution2)
        solution.append(saver)
    solution = torch.tensor(solution)
    return solution

def _plot_mvn():
    pass


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
    pass


def linear_model(
    features: torch.Tensor,
    mu: torch.Tensor,
    sigma: float,
    w: torch.Tensor
) -> torch.Tensor:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    pass


if __name__ == "__main__":
    X, t = load_regression_iris()
    N, D = X.shape
    M, var = 10, 10
    mu = torch.zeros((M, D))
    for i in range(D):
        mmin = torch.min(X[:, i])
        mmax = torch.max(X[:, i])
        mu[:, i] = torch.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, var)
    print(fi)
    print(fi.shape[1])
    print(fi.shape[0])
    print(fi[1,2])

    fig = plt.figure()
    for i in range(fi.shape[0]):
        for j in range(fi.shape[1]):
            # fig = plt(fi[1, j], fi[i,0])
            fig = plt(fi[i], fi[j])
    plt.show()
    _plot_mvn()

    pass
