Section 1:

My Model:
Accuracy of the network on the 10000 test images: 50 %
Accuracy for class: plane is 57.4 %
Accuracy for class: car   is 77.7 %
Accuracy for class: bird  is 31.5 %
Accuracy for class: cat   is 40.9 %
Accuracy for class: deer  is 37.7 %
Accuracy for class: dog   is 38.8 %
Accuracy for class: horse is 57.0 %
Accuracy for class: ship  is 63.6 %
Accuracy for class: truck is 38.5 %

The classification is not that good. The accuracy for some categories are very small (< 50%). This is also be shown by the plot of the missclassification rate. In the beginning it is very unstable. It converges very fast to the end value. For the classes with the higher accuracy, the missclassification rate is smaller.  
confusion matrix:
For classes with small accuracy there is a comparable class. For example the neural network has problems to decide between cats and dogs when a dog picture is given. In comparison it can predict a cat very good when a cat image is given.

Autograd:
Autograd computes the gradient of the parameters in a neural network. With calling the .backprop() the autograd command is called in the background and stores all gradiants.
It is also possible to freeze parameters from which the gradiant should not be calculated. 
In the forward pass it generates a DAG which is used in the backpropagation. 

Section 2:
1. In a RNN (Reccurent Neural Network) the neurons of a layer are not only connected to the following layer (feed forward network). There are also connections between neurons of the same layer or back to neurons from previous layers. It has some kind of a memory.

2. When working with texts the previous letter/word is important for what is comming next. Instead of adding random letters together the RNN calculates the probability of the next letter by using the previous ones. Without the ability of "looking backwards" (output depends on history) it wouldn't be possible to define the next letter/word right.

3. The network output is just a line of random words that sounds/looks like words from that language. There are nor gurantee that this are really some real words. In the example case the output isn't that good. The loss is still at a high value. The quality of a neural networl always depends on the training. 

4. Natural Language Processing, Path planning for robots/machines, Sentiment Classification
