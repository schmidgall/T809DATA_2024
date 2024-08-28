# Author: edit by Sara Schmidgall
# Date: 28.08.2024
# Project: DMML
# Acknowledgements: 
#


from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def gen_data(
    n: int,
    locs: np.ndarray,
    scales: np.ndarray
) -> np.ndarray:
    '''
    Return n data points, their classes and a unique list of all classes, from each normal distributions
    shifted and scaled by the values in locs and scales
    '''
    features = list(())
    targets = list(())
    classes = list(())
    np.set_printoptions(legacy='1.25')

    for i in range(len(locs)):
        featureValue = norm.rvs(locs[i], scales[i], n)
        for k in range(len(featureValue)):
            features.append(featureValue[k])
        for j in range(n):
            targets.append(i)
        classes.append(i)
    return (np.array(features) , np.array(targets) , np.array(classes))


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    sum = 0
    number = 0
    for i in range(len(features)):
        if (targets[i] == selected_class):
            sum += features[i]
            number += 1
    if (number == 0):
        return 0
    return (sum/number)



def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    sum = list(())
    for i in range(len(features)):
        if (targets[i] == selected_class):
            sum.append(features[i])
    if (len(sum) == 0):
        return 0
    sum = np.array(sum)
    return np.cov(sum)
    


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    if (class_cov == 0):
        return 0
    
    probab = list(())
    for value in feature:
        calc = norm.pdf(value, class_mean, class_cov)
        probab.append(calc)
    return np.array(probab)

    


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        ...
    likelihoods = []
    for i in range(test_features.shape[0]):
        ...
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    ...


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    # Section 1
    #gen_data(2, [0, 2], [4, 4])
    #gen_data(1, [-1, 0, 1], [2, 2, 2])
    features, targets, classes = gen_data(25, [-1 , 1], [np.power(5, (1/2)), np.power(5, (1/2))])
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.8)

    # Section 2
    #colors = ["blue", "orange", "yellow", "red", "green", "black"]
    #markers = [".", "x", "+", "*", "1", "p"]
    #for color in classes:
    #    for i in range(len(features)):
    #        if (targets[i] == color):
    #           plt.scatter(features[i], 0, color=colors[color], marker=markers[color])    
    #plt.show()

    # Section 3
    #mean = mean_of_class(train_features, train_targets, 0)

    # Section 4
    #cov = covar_of_class(train_features, train_targets, 0)

    # Section 5
    #class_mean = mean_of_class(train_features, train_targets, 0)
    #class_cov = covar_of_class(train_features, train_targets, 0)
    #probability = likelihood_of_class(test_features[0:3], class_mean, class_cov)

    pass
