import numpy as np

def mean_squared_error(estimates, targets):
    """
    Mean squared error measures the average of the square of the errors (the
    average squared difference between the estimated values and what is 
    estimated. The formula is:

    MSE = (1 / n) * \sum_{i=1}^{n} (Y_i - Yhat_i)^2

    Implement this formula here, using numpy and return the computed MSE

    https://en.wikipedia.org/wiki/Mean_squared_error

    Args:
        estimates(np.ndarray): the estimated values (should be the same shape as targets)
        targets(np.ndarray): the ground truth values

    Returns:
        MSE(int): mean squared error calculated by above equation 
    """
    #calculate size of n data points
    n_data_pt = estimates.size
    #return MSE according to formula
    return 1/n_data_pt * np.sum(np.square(estimates - targets))