import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tools import load_iris


class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()

        '''
        Here we define the layers of the network.
        My SimpleNetwork will take in 5-dimensional
        features and return a 3-dimensional output
        prediction. This could for example be used for
        the Iris dataset which has 4 different features
        dimensions and 3 different flower types.

        I apply softmax on the last layer to get a
        propability distribution over classes
        '''

        self.fc_1 = nn.Linear(4, 100)
        self.fc_2 = nn.Linear(100, 100)
        self.fc_3 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        '''
        This method performs the forward pass,
        x is the input feature being passed through
        the network at the current time
        '''
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return self.softmax(x)


class IrisDataSet(Dataset):
    def __init__(self):
        '''
        A simple PyTorch dataset for the Iris data
        '''

        features, targets, self.classes = load_iris()
        # we first have to convert the numpy data to compatible
        # PyTorch data:
        # * Features should be of type float
        # * Class labels should be of type long
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        '''We always have to define this method
        so PyTorch knows how many items are in our dataset
        '''
        return self.features.shape[0]

    def __getitem__(self, i):
        '''We also have to define this method to tell
        PyTorch what the i-th element should be. In our
        case it's simply the i-th elements from both features
        and targets
        '''
        return self.features[i, :], self.targets[i]


def create_iris_data_loader():
    '''Another convinient thing in PyTorch is the dataloader
    It allows us to easily iterate over all the data in our
    dataset. We can also:
    * set a batch size. In short, setting a batch size > 1
    allows us to train on more than 1 sample at a time and this
    generally decreases training time
    * shuffe the data.
    '''
    dl = DataLoader(IrisDataSet(), batch_size=10, shuffle=True)
    return dl


def train_simple_model():
    # Set up the data
    ds = IrisDataSet()
    dl = DataLoader(ds, batch_size=10, shuffle=True)

    # Initialize the model
    model = SimpleNetwork()

    # Choose a loss metric, cross entropy is often used
    loss_metric = nn.CrossEntropyLoss()
    # Choose an optimizer and connect to model weights
    # Often the Adam optimizer is used
    optimizer = torch.optim.Adam(model.parameters())

    num_steps = 0
    loss_values = []
    # THE TRAINING LOOP
    # we will do 50 epochs (i.e. we will train on all data 50 times)
    for epoch in range(50):
        for (feature, target) in dl:
            num_steps += 1
            # We have to do the following before each
            # forward pass to clear the gradients that
            # we have calculated in the previous step
            optimizer.zero_grad()
            # the prediction output of the network
            # will be a [10 x 3] tensor where out[i,:]
            # represent the class prediction probabilities
            # for sample i.
            out = model(feature)
            # Calculate the loss for the current batch
            loss = loss_metric(out, target)
            # To perform the backward propagation we do:
            loss.backward()
            # The optimizer then tunes the weights of the model
            optimizer.step()

            loss_values.append(loss.mean().item())

    plt.plot(loss_values)
    plt.title('Loss as a function of training steps')
    plt.show()

def train_class_model():
    # define data and normalize
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # defining the network
    net = Net()

    # defining loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    loss_values = []

    # training loop
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # check
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # Accuracy for single class
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    misclass_rate_from = {classname: [] for classname in classes}


    conf_matrix = torch.zeros((len(classes), len(classes)), dtype=torch.int32)

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
                misclass_rate_from[classes[label]].append((total_pred[classes[label]]-correct_pred[classes[label]])/ total_pred[classes[label]])   
                conf_matrix[label, prediction] += 1
            
    plt.imshow(conf_matrix, interpolation='nearest', cmap='Greens')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = torch.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = conf_matrix.max().item() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(conf_matrix[i, j].item(), 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    plt.clf()

    # mis / total
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    print(len(classes))
    for label in range(len(classes)):
        plt.plot(misclass_rate_from[classes[label]], label = classes[label])

    plt.title('misclassification rate')
    plt.xlabel('Iterations')
    plt.ylabel('Misclassification Rate')
    plt.legend()
    plt.show()
    pass



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x





if __name__ == '__main__':
    #train_simple_model()
    train_class_model()
