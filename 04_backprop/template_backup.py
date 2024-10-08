
from typing import Union
import torch
from tools import load_iris, split_train_test
import matplotlib.pyplot as plt


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the sigmoid of x
    '''
    # Handle x < -100 to avoid overflow in e^x
    x_clipped = torch.where(x < -100, torch.tensor(-100.0), x)  # Clip values
    return 1 / (1 + torch.exp(-x_clipped))


def d_sigmoid(x: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    sig = sigmoid(x)  # Calculate the sigmoid of x
    return sig * (1 - sig)

def perceptron(
    x: torch.Tensor,
    w: torch.Tensor
) -> Union[torch.Tensor, torch.Tensor]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    weighted_sum = torch.dot(x, w)  # weighted sum is dot product

    activation_output = sigmoid(weighted_sum)

    return weighted_sum, activation_output

def ffnn(
    x: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    z0 = torch.cat([torch.tensor([1.0]), x])  # (1 x (D+1)) # add bias to input

    a1 = torch.matmul(z0, W1)  # compute input to hidden layer

    z1_no_bias = sigmoid(a1)  # get hidden layer output with sigmoid func
    z1 = torch.cat([torch.tensor([1.0]), z1_no_bias])  # bias to hidden layer

    a2 = torch.matmul(z1, W2)  # input to output layer

    y = sigmoid(a2)  # final output

    return y, z0, z1, a1, a2




def backprop(
    x: torch.Tensor,
    target_y: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    # forward prop
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

    loss = torch.sum(-target_y * torch.log(y))  # cross entropy

    delta_2 = y - target_y  # error from output layer

    delta_1_no_bias = torch.matmul(delta_2, W2[1:, :].T) * (z1[1:] * (1 - z1[1:]))  # error at hidden layer

    # gradients
    dE2 = torch.outer(z1, delta_2)
    dE1 = torch.outer(z0, delta_1_no_bias)

    return y, dE1, dE2


def train_nn(
    X_train: torch.Tensor,
    t_train: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor,
    iterations: int,
    eta: float
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network made and the actual target
    3. Backpropagating the error through the network to adjust the weights.
    '''
    N = X_train.shape[0]  # number of samples
    E_total = []
    misclassification_rate = []
    guesses = torch.zeros(N)

    for i in range(iterations):
        # gradients
        dE1_total = torch.zeros_like(W1)
        dE2_total = torch.zeros_like(W2)

        total_error = 0.0
        misclassifications = 0

        for n in range(N):
            # forward and backprop for every point
            x = X_train[n, :]
            target_y = torch.zeros(K)
            target_y[t_train[n]] = 1.0  # one hot

            # backprop
            y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)

            # sum of gradients
            dE1_total += dE1
            dE2_total += dE2

            # + epsilon bc log(0) = bad
            epsilon = 1e-9
            error = -torch.sum(target_y * torch.log(y + epsilon) + (1 - target_y) * torch.log(1 - y + epsilon))
            total_error += error.item()  # sum of errors

            # check misclassifications
            guess = torch.argmax(y)  # choose class with highest prob
            guesses[n] = guess
            if guess != t_train[n]:
                misclassifications += 1

        # average gradient and update w
        W1 -= eta * dE1_total / N
        W2 -= eta * dE2_total / N

        # total e and misclass for this iteration
        E_total.append(total_error / N)
        misclassification_rate.append(misclassifications / N)

    return W1, W2, torch.tensor(E_total), torch.tensor(misclassification_rate), guesses



def test_nn(
    X: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor
) -> torch.Tensor:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    ...
    N = X.shape[0]
    guesses = torch.zeros(N)  # tensor to store predictions

    for n in range(N):
        x = X[n, :]  # nth data
        # forward prop
        y, _, _, _, _ = ffnn(x, M, K, W1, W2)
        # choose class with highest prob
        guess = torch.argmax(y)
        guesses[n] = guess

    return guesses


if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    # Exercise 1.1
    result = sigmoid(torch.Tensor([0.5]))
    print(result)

    d_result = d_sigmoid(torch.Tensor([0.2]))
    print(d_result)

    # Exercise 1.2
    x1 = torch.Tensor([1.0, 2.3, 1.9])
    w1 = torch.Tensor([0.2, 0.3, 0.1])
    result1 = perceptron(x1, w1)
    print(result1)

    x2 = torch.Tensor([0.2, 0.4])
    w2 = torch.Tensor([0.1, 0.4])
    result2 = perceptron(x2, w2)
    print(result2)  # Expected output: (tensor(0.1800), tensor(0.5449))

    # Exercise 1.3
    # initialize the random generator to get repeatable results
    torch.manual_seed(4321)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = \
        split_train_test(features, targets)

    # initialize the random generator to get repeatable results
    torch.manual_seed(1234)

    # Take one point:
    x = train_features[0, :]
    K = 3  # number of classes
    M = 10
    D = 4
    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

    print("y:", y)
    print("z0:", z0)
    print("z1:", z1)
    print("a1:", a1)
    print("a2:", a2)

    # Exercise 1.4
    # initialize random generator to get predictable results
    torch.manual_seed(42)

    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]

    x = features[0, :]

    # create one-hot target for the feature
    target_y = torch.zeros(K)
    target_y[targets[0]] = 1.0

    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1

    # Perform backpropagation
    y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)

    print("y:", y)
    print("dE1:", dE1)
    print("dE2:", dE2)

    # Section 2
    # Exercise 2.1

    # initialize the random seed to get predictable results
    torch.manual_seed(1234)

    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]

    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
        train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)

    print("W1tr:", W1tr)
    print("W2tr:", W2tr)
    print("Etotal:", Etotal)
    print("misclassification_rate:", misclassification_rate)
    print("last_guesses:", last_guesses)

    # Exercise 2.2
    guesses = test_nn(test_features, M, K, W1tr, W2tr)

    print("Guesses:", guesses)

    # Exercise 2.3
    # set a unique seed
    seed = 1111  # Replace with your unique seed
    torch.manual_seed(1234)

    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]

    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
        train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)


    # test network
    test_guesses = test_nn(test_features, M, K, W1_trained, W2_trained)

    # accuracy calc
    accuracy = torch.sum(test_guesses == test_targets) / len(test_targets)
    print(f"Accuracy: {accuracy.item() * 100:.2f}%")

    # confusion matrix
    num_classes = len(classes)
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)

    for i in range(len(test_targets)):
        true_class = int(test_targets[i].item())  # Convert to int
        predicted_class = int(test_guesses[i].item())  # Convert to int
        conf_matrix[true_class, predicted_class] += 1

    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap='Greens')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = torch.arange(num_classes)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = conf_matrix.max().item() / 2
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(conf_matrix[i, j].item(), 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    # e total plot
    plt.figure()
    plt.plot(E_total.numpy())
    plt.title('E_total over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Total Error')
    plt.show()

    # misclass rate
    plt.figure()
    plt.plot(misclassification_rate.numpy())
    plt.title('misclassification_rate over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Misclassification Rate')
    plt.show()
