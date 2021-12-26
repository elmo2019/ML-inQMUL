import numpy as np

from load_data_ex2 import *
from normalize_features import *
from gradient_descent import *
from calculate_hypothesis import *
import os

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# This loads our data
X, y = load_data_ex2()

# Normalize
X_normalized, mean_vec, std_vec = normalize_features(X)

# After normalizing, we append a column of ones to X, as the bias term
column_of_ones = np.ones((X_normalized.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
X_normalized = np.append(column_of_ones, X_normalized, axis=1)

# initialise trainable parameters theta, set learning rate alpha and number of iterations
theta = np.zeros((3))
alpha = 0.01
iterations = 200

# plot predictions for every iteration?
do_plot = True

# call the gradient descent function to obtain the trained parameters theta_final
theta_final = gradient_descent(X_normalized, y, theta, alpha, iterations, do_plot)
print("print the final theta")
print(theta_final)

#########################################
# Write your code here
# Create two new samples: (1650, 3) and (3000, 4)
# Calculate the hypothesis for each sample, using the trained parameters theta_final
# Make sure to apply the same preprocessing that was applied to the training data
# Print the predicted prices for the two samples
samples_1 = np.array([1650,3])
samples_2 = np.array([3000,4])
samples_1_normalized = (samples_1 - mean_vec) / std_vec
samples_2_normalized = (samples_2 - mean_vec) / std_vec
column_of_ones = np.ones((samples_1_normalized.shape[0], 1))
samples_1_normalized = np.append(column_of_ones, samples_1_normalized, axis=1)
column_of_ones = np.ones((samples_2_normalized.shape[0], 1))
samples_2_normalized = np.append(column_of_ones, samples_2_normalized, axis=1)
hypothesis_sample_1 = theta_final * samples_1_normalized
hypothesis_sample_1 = np.sum(hypothesis_sample_1)
hypothesis_sample_2 = np.sum(theta_final * samples_2_normalized)

print("the prediction prices for sample_1 is",hypothesis_sample_1)
print("the prediction prices for sample_2 is",hypothesis_sample_2)
########################################/
