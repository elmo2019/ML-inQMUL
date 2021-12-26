import numpy as np

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    hypothesis=0
    #########################################
    # Write your code here
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.
    #hypothesis=theta[0]+theta[1]*X[i,0]+theta[2]*X[i,1]
    hypothesis = theta * X[i,:]
    hypothesis = sum(hypothesis)
    ########################################/
    
    return hypothesis
