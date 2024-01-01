import numpy as np 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size # define input size
        self.hidden_size = hidden_size # define hidden size
        self.output_size = output_size # define output size

        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size) # define input to hidden weights
        self.bias_hidden = np.zeros((1, self.hidden_size)) # define bias input to hidden 
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)# define hidden to output weights 
        self.bias_output = np.zeros((1, self.output_size))# define bias hidden to output 

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x * lambda_learning_rate)) #sigmoid equation/activation function 

    def sigmoid_derivative(self, y):          
        return y * (1 - y) #derivative function used in back propagation

    def forward(self, input_data):
        # Forward propagation
        self.input_data = input_data # get input data 
        self.hidden_layer_input = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden # taking input values and weights and mulitplying them for the hidden value 
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input) # using the activation function on the hidden layer to get the new values 
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output # taking new hidden values and weights from output weights and mulitplying them for the output value 
        self.output = self.sigmoid(self.output_layer_input)  # using the activation function on the output layer to get the new values 
        return self.output # returing new output values

    def backward(self, target, lambda_learning_rate, eta_learning_rate):
        # Backpropagation
        error = target - self.output # defining the error
        
        # Output layer
        output_delta = lambda_learning_rate * error * self.sigmoid_derivative(self.output) # defining the output delta 
        hidden_layer_output_transpose = self.hidden_layer_output.T # transposing the hidden layer output so that it matches up with the output delta shape for the np.dot function
        self.weights_hidden_output += np.dot(hidden_layer_output_transpose, output_delta) * eta_learning_rate # getting new weight values for the output to hidden values
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * eta_learning_rate # updating the bias weights in the output layer based on the gradient of the error with respect to the output.

        # updating the weights between input and hidden layer0
        hidden_layer_error = np.dot(output_delta, self.weights_hidden_output.T) # added hidden layer error 
        hidden_layer_delta = lambda_learning_rate * hidden_layer_error * self.sigmoid_derivative(self.hidden_layer_output)
        input_data_transpose = self.input_data.T
        self.weights_input_hidden += np.dot(input_data_transpose, hidden_layer_delta) * eta_learning_rate
        self.bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * eta_learning_rate

    def train(self, input_data, target, eta_learning_rate, lambda_learning_rate, epochs):
        for _ in range(epochs):
            self.forward(input_data)
            self.backward(target, eta_learning_rate, lambda_learning_rate)

    def predict(self, input_data):
        return self.forward(input_data)
    

# Loading data from Excel sheet
data = pd.read_csv('ce889_dataCollection.csv')

# Assign columns to the data
data.columns = ['pos_x', 'pos_y', 'vel_y', 'vel_x']

# Remove rows with null or zero values
data = data.dropna()  # Remove rows with null values
data = data[(data != 0).all(1)]  # Remove rows with all zero values

# normalize the data
normalized_data = pd.DataFrame()

for column in data.columns:
    min_val = data[column].min()
    max_val = data[column].max()
    normalized_data[column] = (data[column] - min_val) / (max_val - min_val)

# Finding min and max values for the NeuralNetHolder
min_pos_x = data['pos_x'].min()
max_pos_x = data['pos_x'].max()

min_pos_y = data['pos_y'].min()
max_pos_y = data['pos_y'].max()

min_vel_x = data['vel_x'].min()
max_vel_x = data['vel_x'].max()

min_vel_y = data['vel_y'].min()
max_vel_y = data['vel_y'].max()

print(min_pos_x, max_pos_x, min_pos_y, max_pos_y, min_vel_x, max_vel_x, min_vel_y, max_vel_y)

# Define the input columns and target columns based on the data
input_columns = ['pos_x', 'pos_y']  # input is the position
target_columns = ['vel_x', 'vel_y']  # target is the velocity

# Convert the input and target data to NumPy arrays for manipulation reasons
input_data = data[input_columns].values
target_data = data[target_columns].values
print(target_data)

# setting up training 
x_train, x_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.2, random_state=42)
# checking length 
input_size = len(input_columns)
output_size = len(target_columns)


# Define and train the neural network using sizes, learning rate and epochs
hidden_size = 10
lambda_learning_rate = 0.05
eta_learning_rate = 0.05
epochs = 100

nn = NeuralNetwork(input_size, hidden_size, output_size)    
RMSE_values = []

for epoch in range(epochs):
    total_error = 0  # Initialize total error for this epoch

    # Loop through all rows in the dataset
    for i in range(len(x_train)):
        input_sample = x_train[i].reshape(1, -1)  # Reshaping the input data to (1, 4)
        target_sample = y_train[i]  # No need to reshape the target data 
        nn.train(input_sample, target_sample, lambda_learning_rate, eta_learning_rate, 1)

        # Calculating the error for this sample and adding it to the total error
        sample_error = np.mean(np.square(target_sample - nn.predict(input_sample)))
        total_error += sample_error
    # finding the RMSE for the training data
    MSE_train = total_error / len(x_train)
    RMSE_train = math.sqrt(MSE_train)

    # Make predictions on the testing set 
    predictions_test = nn.predict(x_test)

    # Calculate RMSE for testing set 
    total_error_test = np.mean(np.square(y_test - predictions_test))
    RMSE_test = math.sqrt(total_error_test)

    # Appending the RMSE values for comparison 
    RMSE_values.append((RMSE_train, RMSE_test))

    # Print RMSE values for each epoch
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Training RMSE: {RMSE_train}, Testing RMSE: {RMSE_test}")

    
# putting the trained weights into pickle so we can use them in the game 
model_par = {
    'hidden_weights': nn.weights_hidden_output,
    'input_weights': nn.weights_input_hidden,
    'bias_hidden': nn.bias_hidden,
    'bias_output' : nn.bias_output
}
with open('model.pk1', 'wb') as file:
    pickle.dump(model_par, file)

# Make predictions on the entire dataset after training
predictions = nn.predict(input_data)
print("Final Predictions:")
print(predictions)

epochs_list = range(1, epochs + 1)
RMSE_train_values = [item[0] for item in RMSE_values]
RMSE_test_values = [item[1] for item in RMSE_values]

#plotting graph
plt.plot(epochs_list, RMSE_train_values, label='Training RMSE')
plt.plot(epochs_list, RMSE_test_values, label='Testing RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('RMSE Comparison between Training and Testing')
plt.legend()
plt.show()