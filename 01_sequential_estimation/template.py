# Author: Sara Schmidgall
# Date: 27.08.2024
# Project: DMML
# Acknowledgements: 
#

import matplotlib.pyplot as plt
import numpy as np

from tools import scatter_2d_data, bar_per_axis


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: np.float64
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    covariance = np.power(var, 2) * np.identity(k)
    data = np.empty([n, k])
    for i in range(n):
        data[i] = np.random.multivariate_normal(mean, covariance)
    return data


def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    return mu + (1/n) * (x - mu)


def _plot_sequence_estimate():
    data = gen_data(100, 2, np.array([0,0]), 3)
    estimates = [np.array([0, 0])]
    for i in range(data.shape[0]):
        new_estimate = update_sequence_mean(estimates[-1], data[i], i+1)
        estimates.append(new_estimate)
            
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')

    plt.legend(loc='upper center')
    #plt.show()


def _square_error(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    return np.power((y - y_hat), 2)



def _plot_mean_square_error():
    data = gen_data(100, 2, np.array([0,0]), 3)
    estimates = [np.array([0, 0])]
    meanSqrErr = list(())

    for i in range(data.shape[0]):
        new_estimate = update_sequence_mean(estimates[-1], data[i], i+1)
        estimates.append(new_estimate)

        errorSqrt = _square_error(np.mean(data, 0), new_estimate)
        meanSqrErr.append(np.mean(errorSqrt))
    
    plt.plot(meanSqrErr)
    plt.show()


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: np.float64
) -> np.ndarray:
    # remove this if you don't go for the independent section
    pass


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    pass


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    # Section 1
    #X = gen_data(2, 3, np.array([0, 1, -1]), 1.3)
    #X = gen_data(5, 1, np.array([0.5]), 0.5)

    # Section 2
    #dataPoints = gen_data(300, 2, np.array([-1, 2]), np.power(4,1/2))
    #scatter_2d_data(dataPoints)
    #bar_per_axis(dataPoints)

    # Section 3
    #X = gen_data(300, 2, np.array([-1, 2]), np.power(4,1/2))
    #mean = np.mean(X, 0)
    #new_x = gen_data(1, 2, np.array([0, 0]), 1)
    #update_sequence_mean(mean, new_x, X.shape[0]+1)

    # Section 4
    #_plot_sequence_estimate()

    # Section 5
    _plot_mean_square_error()

