import numpy as np


class Loss:
    """
    An abstract base class for a loss function that computes both the prescribed
    loss function (the forward pass) as well as its gradient (the backward
    pass).

    *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

    Arguments:
        regularization - (`Regularization` or None) The type of regularization to
            perform. Either a derived class of `Regularization` or None. If None,
            no regularization is performed.
    """

    def __init__(self, regularization=None):
        self.regularization = regularization

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss.

        *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The calculated loss normalized by the number of
                examples, N.
        """
        pass

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        pass


class SquaredLoss(Loss):
    """
    The squared loss function.
    """

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss. The squared loss for a single example
        is given as follows:

        L_s(x, y; w) = (1/2) (y - w^T x)^2

        The squared loss over a dataset of N points is the average of this
        expression over all N examples.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The calculated loss normalized by the number of
                examples, N.
        """
        
        x_shape = X.shape[0]
        x_copy = X.copy()
        loss_array = (y - np.array([np.dot(w.T, i, out=None) for i in x_copy]))
        loss_sq = np.square(loss_array)
        l_abs = np.abs(loss_array)
        loss_scaler = np.sum(loss_sq)*0.5        


        if self.regularization == True:
            return loss_scaler/x_shape + self.regularization.forward(w)
        else:
            return loss_scaler/x_shape

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        
        x_shape = X.shape[0]
        gradient = np.zeros((X.shape))
        x_copy = X.copy()

        g_array = (y - np.array([np.dot(w.T, i, out=None) for i in x_copy]))
        g_sub = np.array([g_array[i] * X[i, :] for i in range(x_shape)])
        g_scaler = np.sum(g_sub,0)
        gradient_scaler = g_scaler*(-1)
        
        if self.regularization == True:
            return gradient_scaler/x_shape + self.regularization.backward(w)
        else:
            return gradient_scaler/x_shape
        
class HingeLoss(Loss):
    """
    The hinge loss function.

    https://en.wikipedia.org/wiki/Hinge_loss
    """

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss. The hinge loss for a single example
        is given as follows:

        L_h(x, y; w) = max(0, 1 - y w^T x)

        The hinge loss over a dataset of N points is the average of this
        expression over all N examples.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The calculated loss normalized by the number of
                examples, N.
        """
        
        x_shape = X.shape[0]
        loss = np.zeros((x_shape))
        x_copy = X.copy()

        l_array =  np.array([np.dot(w.T, i, out=None) for i in x_copy])
        l_mul = 1 - y*l_array*1
        l_sub = np.where(l_mul < 0, 0, l_mul)
        l_sort = np.sort(l_mul)
        loss_scaler = np.sum(l_sub)*1

        if self.regularization == True:
            return loss_scaler/x_shape + self.regularization.forward(w)
        else:
            return loss_scaler/x_shape

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        
        x_shape = X.shape[0]
        gradient = np.zeros((X.shape), dtype='float')

        for i in range(x_shape):
            g_stack = np.vstack(X[i, :])
            g_mat = (1 - y[i] * np.atleast_2d(w) @ np.vstack(X[i, :]) > 0)
            #print(g_mat)
            gradient[i, :] =  g_mat* y[i] * (-1) * X[i, :]

        gradient_scaler = np.sum(gradient, 0)*(1)

        if self.regularization == True:
            return gradient_scaler / x_shape + self.regularization.backward(w)
        else:
            return gradient_scaler / x_shape
        


class ZeroOneLoss(Loss):
    """
    The 0-1 loss function.

    The loss is 0 iff w^T x == y, else the loss is 1.

    *** YOU DO NOT NEED TO IMPLEMENT THIS ***
    """

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss. The squared loss for a single example
        is given as follows:

        L_0-1(x, y; w) = {0 iff w^T x == y, else 1}

        The squared loss over a dataset of N points is the average of this
        expression over all N examples.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The average loss.
        """
        predictions = (X @ w > 0.0).astype(int) * 2 - 1
        loss = np.sum((predictions != y).astype(float)) / len(X)
        if self.regularization:
            loss += self.regularization.forward(w)
        return loss

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        # This function purposefully left blank
        raise ValueError('No need to use this function for the homework :p')
