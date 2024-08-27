import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: np.float64, mu: np.float64) -> np.ndarray:
    exponent = - (np.power((x-mu), 2)) / (2* np.power(sigma, 2))
    denominator = np.power((2*np.pi*np.power(sigma, 2)), -1/2)
    return (denominator * np.exp(exponent))
    # Part 1.1

def plot_normal(sigma: np.float64, mu:np.float64, x_start: np.float64, x_end: np.float64):
    x_range = np.linspace(x_start, x_end, 500)
    plt.plot(x_range,normal(x_range, sigma, mu), label=f'mu={mu}, sigma={sigma}')
    plt.legend(loc='upper left')
    # Part 1.2

def _plot_three_normals():
    plt.clf()
    plot_normal(0.5, 0, -5, 5)
    plot_normal(0.25, 1, -5, 5)
    plot_normal(1, 1.5, -5, 5)
    plt.legend(loc='upper left')
    # Part 1.2

def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    sum = 0
    for i in range(len(sigmas)):
        exponent = - (np.power((x-mus[i]), 2)) / (2* np.power(sigmas[i], 2))
        denominator = np.power((2*np.pi*np.power(sigmas[i], 2)), -1/2)
        sum = sum + ((weights[i] * denominator) * np.exp(exponent))
    return (sum)
    # Part 2.1

def _compare_components_and_mixture():
    print ("Hello world 5")
    # Part 2.2

def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    print ("Hello world 6")
    # Part 3.1

def _plot_mixture_and_samples():
    print ("Hello world 7")
    # Part 3.2

if __name__ == '__main__':
    # Part 1.2
    #plot_normal(0.5, 0, -2, 2)
    #plt.show()
    #_plot_three_normals()

    # Part 2.1
    #normal_mixture(np.linspace(-5, 5, 5), [0.5, 0.25, 1], [0, 1, 1.5], [1/3, 1/3, 1/3])
    #normal_mixture(np.linspace(-2, 2, 4), [0.5], [0], [1])

    # Part 2.2
    plt.clf()
    plot_normal(0.5, 0, -5, 5)
    plot_normal(1.5, -0.5, -5, 5)
    plot_normal(1.5, 0.25, -5, 5)
    mix = normal_mixture(np.linspace(-5, 5, 5), [0.5, 1.5, 1.5], [0, -0.5, 0.25], [1/3, 1/3, 1/3])
    plt.show()


    # select your function to test here and do `python3 template.py`
