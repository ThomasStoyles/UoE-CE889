import numpy as np 
import pickle
import pandas as pd 
class NeuralNetHolder:

    def __init__(self):
        self.input_size = 2 # define input size
        self.hidden_size = 10 # define hidden size
        self.output_size = 2 # define output size

        # Initialize weights and biases

        loading_paras = pickle.load(open('model.pk1', 'rb'))
                
        load_hw = loading_paras['hidden_weights']
        self.load_hw = load_hw
        load_iw = loading_paras['input_weights']
        self.load_iw = load_iw
        load_Hbias = loading_paras['bias_hidden']
        self.hbias = load_Hbias
        load_Ibias = loading_paras['bias_output']
        self.Ibias = load_Ibias
    
    def sigmoid(self, x, lambda_learning_rate = 0.05):
        return 1 / (1 + np.exp(-x * lambda_learning_rate)) #sigmoid equation/activation function 

    def sigmoid_derivative(self, y):
        return y * (1- y) #derivative function used in back propagation

    def forward(self, input_data):
        # Forward propagation
        self.input_data = input_data # get input data 
        self.hidden_layer_input = np.dot(input_data, self.load_iw) + self.hbias # taking input values and weights and mulitplying them for the hidden value 
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input) # using the activation function on the hidden layer to get the new values 
        self.output_layer_input = np.dot(self.hidden_layer_output, self.load_hw) + self.Ibias # taking new hidden values and weights from output weights and mulitplying them for the output value 
        self.output = self.sigmoid(self.output_layer_input)  # using the activation function on the output layer to get the new values 
        return self.output # returing new output values



    def predict(self, input_row):
 
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X VEL AND Y VEL
        self.input_data = input_row
        print(input_row)

        location = list(map(float, input_row.split(',')))
        print(location)

        pos_x_min = -633.5556847753094
        pos_x_max = 619.3352181554503
        pos_y_min = 66.11491246515175
        pos_y_max = 376.28776538501256

        # Normalization
        x_norm = (location[0] - pos_x_min) / (pos_x_max - pos_x_min)
        y_norm = (location[1] - pos_y_min) / (pos_y_max - pos_y_min)


        # using normalized distances for prediction 
        location = [x_norm, y_norm]
        print(location)
        neural_net = NeuralNetHolder()
        predict = neural_net.forward([x_norm, y_norm])
        print(predict)
        print(location)

        predict = predict.flatten()
        predict = predict.tolist()

        vel_y_min = -3.015536943160625
        vel_y_max =  3.409270532113017
        vel_x_min = -1.8914137330894876
        vel_x_max = 7.499999999999989
    
        load_pred_output_denorm = [predict[0] * (vel_x_max - vel_x_min) + vel_x_min,
                                predict[1] * (vel_y_max - vel_y_min) + vel_y_min]
        return load_pred_output_denorm