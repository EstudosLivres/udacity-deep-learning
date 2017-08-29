## data_prep.py
import os
import numpy as np
import pandas as pd

admissions = pd.read_csv(os.getcwd() + '/2-13-binary.csv')

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:, field] = (data[field] - mean) / std

# Split off random 10% of the data for testing
np.random.seed(21)
sample = np.random.choice(data.index, size=int(len(data) * 0.9), replace=False)
data, test_data = data.loc[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

## backprop.py
import numpy as np

# from data_prep import features, targets, features_test, targets_test
np.random.seed(21)


# Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900  # loop interactives
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None

# Initialize random weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    # delta weights from hidden input
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    # delta weights to hidden output
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    # retrieving data from csv (data_prep.py)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # TODO: Calculate the output
        hidden_input = np.dot(x, weights_input_hidden)  # x vs weights_input_hidden LINEAR COMBINATION
        hidden_output = sigmoid(hidden_input)  # LINEAR COMBINATION (hidden_input) activate by sigmoid (activation function)
        outputs_linear_combination = np.dot(hidden_output, weights_hidden_output)
        output = sigmoid(outputs_linear_combination)

        ## Backward pass ##
        # TODO: Calculate the network's prediction error
        error = y - output  # y -y^

        # TODO: Calculate error term for the output unit (output_gradient)
        output_error_term = error * output * (1 - output)  # DONE: sigmoid_prime math (derivate)

        ## propagate errors to hidden layer

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term,
                              weights_hidden_output)  # output_error & hidden_wrights_out LinearCombination

        # TODO: Calculate the error term for the hidden layer (hidden_gradient)
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)  # DONE: sigmoid_prime math (derivate)

        # TODO: Update the change in weights
        x_as_column_vector = x[:, None]

        # output_error_term.shape  =  escalar()   #output_grad
        # hidden_output.shape      =  vector(2,)  #hidden_outputs

        # hidden_error_term.shape  =  vector(2,)  #hidden_grad
        # x.shape                  =  vector(6,)  #inputs
        # x_as_column_vector.shape =  matriz(6,1) #inputs_refactored

        del_w_hidden_output += output_error_term * hidden_output  # ∑ wi (weights) * xi (inputs)
        del_w_input_hidden += hidden_error_term * x_as_column_vector

    # TODO: Update weights (skipping this step the accuracy would be too low)
    # UPDATE Weight: n*(y-y^)*f'(h) = n_records = f'(h) = 1/f(h)
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and loss > last_loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))