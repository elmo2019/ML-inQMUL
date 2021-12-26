import numpy as np
from load_data_ex1 import *
from normalize_features import *
from gradient_descent import *
from plot_data_function import *
from plot_boundary import *
import matplotlib.pyplot as plt
import os

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# This loads our data
X, y = load_data_ex1()


# Create the features x1*x2, x1^2 and x2^2
#########################################
# Write your code here
# Compute the new features
# Insert extra singleton dimension, to obtain Nx1 shape
# Append columns of the new features to the dataset, to the dimension of columns (i.e., 1)
"""""
columns_x1x2 = np.array([], dtype=np.float32)
tempt_array= X[:,2]*np.transpose(X[:,1])
print(tempt_array)
columns_x1x2 = np.append(columns_x1x2,)
X = np.append(columns_x1x2,X,axis=1)

columns_doublex1 = np.array([], dtype=np.float32)
columns_doublex1 = np.append(columns_doublex1,X[:,1]*X[:,1])
X = np.append(columns_doublex1,X,axis=1)

columns_doublex2 =np.array([], dtype=np.float32)
columns_doublex2 = np.append(columns_doublex2,X[:,1]*X[:,2])
X = np.append(columns_doublex2,X,axis=1)
???? why wrong?
"""
#写一个for循环,新建一些列表
columns_x1x2 = np.array([], dtype=np.float32)
columns_doublex1 = np.array([], dtype=np.float32)
columns_doublex2 =np.array([], dtype=np.float32)
for i in range(X.shape[0]):
    curren_Value = X[i,0]*X[i,1]
    columns_x1x2 = np.append(columns_x1x2,curren_Value)

    curren_Value = X[i,0]*X[i,0]
    columns_doublex1 = np.append(columns_doublex1,curren_Value)

    curren_Value = X[i,1]*X[i,1]
    columns_doublex2 = np.append(columns_doublex2,curren_Value)
X=np.c_[X,columns_x1x2]
X=np.c_[X,columns_doublex1]
X=np.c_[X,columns_doublex2]
########################################/

# Normalize
X_normalized, mean_vec, std_vec = normalize_features(X)

# After normalizing, we append a column of ones to X_normalized, as the bias term
column_of_ones = np.ones((X_normalized.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
X_normalized = np.append(column_of_ones, X_normalized, axis=1)

# Initialise trainable parameters theta
#########################################
# Write your code here
theta = np.zeros((6))
########################################/

# Set learning rate alpha and number of iterations
alpha = 0.1
iterations = 100

# Call the gradient descent function to obtain the trained parameters theta_final and the cost vector
theta_final, cost_vector = gradient_descent(X_normalized, y, theta, alpha, iterations)

# Plot the cost for all iterations
fig, ax1 = plt.subplots()
plot_cost(cost_vector, ax1)
plot_filename = os.path.join(os.getcwd(), 'figures', 'ex3_cost.png')
plt.savefig(plot_filename)
min_cost = np.min(cost_vector)
argmin_cost = np.argmin(cost_vector)
print('Final cost: {:.5f}'.format(cost_vector[-1]))
print('Minimum cost: {:.5f}, on iteration #{}'.format(min_cost, argmin_cost+1))

# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()
