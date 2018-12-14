import numpy
#scipy.special for the sigmoid function expit()
import scipy.special
#library for plotting arrays
import matplotlib.pyplot
import scipy.misc
#scipy misc必须引用pillow才能用

# nerural network class definition
class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        # set number of nodes in each input ,hideden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # set learning rate
        self.lr = learningrate

        # weights权重
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        #activation function is the sigmoid function
        self.activation_functiion = lambda  x:scipy.special.expit(x)
        pass

    # train the neural network
    def train(self,inputs_list,targets_list):
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)
        #calculate signals emerging from hidden layer
        hidden_outputs = self.activation_functiion(hidden_inputs)
        #calculate signals into final putput layer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        #calculate signals emerging from final output layer
        final_outputs = self.activation_functiion(final_inputs)

        #output layer error is the (target-final_outputs)
        output_errors = targets - final_outputs
        #hidden layer error is the output_errors,split by weights,recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T,output_errors)

        #update the weights for the links between the hidden and output layer
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0-final_outputs))
                                        ,numpy.transpose(hidden_outputs))
        #update the weights for the links between the input and hidden layer
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *(1.0-hidden_outputs))
                                       ,numpy.transpose(inputs))
        pass

    # query the neural network
    def query(self,inputs_list):
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list,ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)

        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_functiion(hidden_inputs)

        #calculate the signals into final output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)

        #calculate the signals emerging from final putput layer
        final_ouputs = self.activation_functiion(final_inputs)

        return final_ouputs
#number of input,hidden and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

#learning rate
learning_rate = 0.2

#create instance of  neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#load the mnist training data CSV file into a list
training_data_file = open("mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#train the neural network
#go through all records in the training data set
fleg = 0
#epochs is the number of times the training data set is used for training
epochs = 6
for e in range(epochs):
    for record in training_data_list[1:100]:
        #split the record by the ','commas
        all_values = record.split(',')
        #scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        #create the target output values (all 0.01,except the desired label which is 0.99)
        target = numpy.zeros(output_nodes)+0.01
        #all_values[0] is the target label for this record
        target[int(all_values[0])] = 0.99
        n.train(inputs,target)
        fleg+=1
        print(fleg)
        pass
    pass
#load the mnist test data CSV file into a list
test_data_file = open("mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

img_array = scipy.misc.imread("22.jpg",flatten=True)
img_data = 255.0-img_array.reshape(784)
img_data = (img_data/255.0*0.99)+0.01
#image_a = numpy.asfarray(img_data).reshape((28,28))
#inputs = (image_a / 255.0 * 0.99) + 0.01
# query the network
outputs = n.query(img_data)
# the index of the highest value corresponds to the label
label = numpy.argmax(outputs)
print(label, "network's answer")

#test the neural network
#scorecard for how well the network performs ,initially empty
scorecard = []
#go through all the records in the test data set
for record in test_data_list:
    #split the record by the ','commas
    all_values = record.split(',')
    #correct answer is first value
    correct_label = int(all_values[0])
    print(correct_label,"correct_label")
    #scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    #query the network
    outputs = n.query(inputs)
    #the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print(label,"network's answer")
    #append correct or incorrect to list
    if(label == correct_label):
        #network's answer matches correct answer ,add 1 to scorecard
        scorecard.append(1)
    else:
        #network's answer doesn't match correct answer,add 0 to scorecard
        scorecard.append(0)
        pass
    pass
scorecard_array = numpy.asarray(scorecard)
print("performance = ",scorecard_array.sum()/scorecard_array.size)