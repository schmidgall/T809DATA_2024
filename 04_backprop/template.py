from typing import Union
import torch
import matplotlib.pyplot as plt
from tools import load_iris, split_train_test


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the sigmoid of x
    '''
    x2 =  torch.where(x < -100, torch.tensor(-100.0), x)
    return (1/ (1 + torch.exp(-x2)))

def d_sigmoid(x: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    return (sigmoid(x) * (1-sigmoid(x)))


def perceptron(
    x: torch.Tensor,
    w: torch.Tensor
) -> Union[torch.Tensor, torch.Tensor]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    sum = 0
    for i in range(len(x)):
        sum = sum + (x[i] * w[i])
    return (sum, sigmoid(sum))

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
    z0 = torch.cat([torch.tensor([1.0]), x]) 
    a1 = torch.matmul(z0, W1)  
    z1_no_bias = sigmoid(a1)  
    z1 = torch.cat([torch.tensor([1.0]), z1_no_bias]) 
    a2 = torch.matmul(z1, W2)
    y = sigmoid(a2)

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
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

    delta_2 = y - target_y 

    delta_1_no_bias = torch.matmul(delta_2, W2[1:, :].T) * (z1[1:] * (1 - z1[1:]))  # error at hidden layer

    dE1 = torch.zeros(W1.shape)
    dE2 = torch.zeros(W2.shape)

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
            target_y[t_train[n]] = 1.0

            # backprop
            y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)

            # sum of gradients
            dE1_total += dE1
            dE2_total += dE2

            epsilon = 1e-9
            error = -torch.sum(target_y * torch.log(y + epsilon) + (1 - target_y) * torch.log(1 - y + epsilon))
            total_error += error.item()  

            # check misclassifications
            guess = torch.argmax(y) 
            guesses[n] = guess
            if guess != t_train[n]:
                misclassifications += 1

        W1 -= eta * dE1_total / N
        W2 -= eta * dE2_total / N
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
    n = X.shape[0]
    guesses = torch.zeros(n)

    for i in range(n):
        x = X[i, :]
        y, _, _, _, _ = ffnn(x, M, K, W1, W2)
        guess = torch.argmax(y)
        guesses[i] = guess

    return guesses


if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    # # Part 1.1
    # sigfct = sigmoid(torch.Tensor([0.5]))
    # derivative = d_sigmoid(torch.Tensor([0.2]))

    # solution = perceptron(torch.Tensor([1.0, 2.3, 1.9]), torch.Tensor([0.2, 0.3, 0.1]))
    # solitoon2 = perceptron(torch.Tensor([0.2, 0.4]), torch.Tensor([0.1, 0.4]))
    # print(solution)
    # print(solitoon2)

    # # Part 1.3
    # # initialize the random generator to get repeatable results
    # torch.manual_seed(4321)
    
    # features, targets, classes = load_iris() 
    # (train_features, train_targets), (test_features, test_targets) = \
    #     split_train_test(features, targets)

    # # Take one point:
    # x = train_features[0, :]
    # K = 3  # number of classes (output neurons)
    # M = 10 # nbr hidden layer neurons
    # D = 4 # nbr of inputs

    # # Initialize two random weight matrices
    # W1 = 2 * torch.rand(D + 1, M) - 1
    # W2 = 2 * torch.rand(M + 1, K) - 1

    # y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

    # # Part 1.4
    # # initialize random generator to get predictable results
    # torch.manual_seed(42)

    # K = 3  # number of classes
    # M = 6
    # D = train_features.shape[1]

    # x = features[0, :]

    # # create one-hot target for the feature
    # target_y = torch.zeros(K)
    # target_y[targets[0]] = 1.0

    # # Initialize two random weight matrices
    # W1 = 2 * torch.rand(D + 1, M) - 1
    # W2 = 2 * torch.rand(M + 1, K) - 1

    # y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)

    # Part 2.1
    # initialize the random seed to get predictable results
    # torch.manual_seed(1234)

    # K = 3  # number of classes
    # M = 6
    # D = train_features.shape[1]

    # Initialize two random weight matrices
    # W1 = 2 * torch.rand(D + 1, M) - 1
    # W2 = 2 * torch.rand(M + 1, K) - 1
    # W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)


    # Part 2.3
    torch.manual_seed(3455)
    features, targets, classes = load_iris() 
    
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.8)

    K = 5 # Nbr classes
    M = 10  # hidden layer
    D = train_features.shape[1]  # Nbr input features

    # random weights
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1

    # train network
    iterations = 700
    learning_rate = 0.1
    W1_trained, W2_trained, E_total, misclassification_rate, _ = train_nn(
        train_features, train_targets, M, K, W1, W2, iterations, learning_rate
    )

    test_guesses = test_nn(test_features, M, K, W1_trained, W2_trained)
    # accuracy calc
    accuracy = torch.sum(test_guesses == test_targets) / len(test_targets)
    print(f"Accuracy: {accuracy.item() * 100:.2f}%")

    # confusion matrix
    num_classes = len(classes)
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)

    for i in range(len(test_targets)):
        true_class = int(test_targets[i].item()) 
        predicted_class = int(test_guesses[i].item())
        conf_matrix[true_class, predicted_class] += 1

    # plot confusion matrix
    plt.figure(figsize=(5, 10))
    plt.imshow(conf_matrix, interpolation='nearest')
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
    plt.title('E_total')
    plt.xlabel('Iterations')
    plt.ylabel('Total Error')
    plt.show()

    # misclass rate
    plt.figure()
    plt.plot(misclassification_rate.numpy())
    plt.title('misclassification rate')
    plt.xlabel('Iterations')
    plt.ylabel('Misclassification Rate')
    plt.show()


