import numpy as np
from your_code import HingeLoss, SquaredLoss
from your_code import L1Regularization, L2Regularization
from .metrics import accuracy

class GradientDescent:
    """
    This is a linear classifier similar to the one you implemented in the
    linear regressor homework. This is the classification via regression
    case. The goal here is to learn some hyperplane, y = w^T x + b, such that
    when features, x, are processed by our model (w and b), the result is
    some value y. If y is in [0.0, +inf), the predicted classification label
    is +1 and if y is in (-inf, 0.0) the predicted classification label is
    -1.

    The catch here is that we will not be using the closed form solution,
    rather, we will be using gradient descent. In your fit function you
    will determine a loss and update your model (w and b) using gradient
    descent. More details below.

    Arguments:
        loss - (string) The loss function to use. Either 'hinge' or 'squared'.
        regularization - (string or None) The type of regularization to use.
            One of 'l1', 'l2', or None. See regularization.py for more details.
        learning_rate - (float) The size of each gradient descent update step.
        reg_param - (float) The hyperparameter that controls the amount of
            regularization to perform. Must be non-negative.
    """
    def __init__(self, loss, regularization=None,
                 learning_rate=0.01, reg_param=0.05):
        self.learning_rate = learning_rate

        # Select regularizer
        if regularization == 'l1':
            regularizer = L1Regularization(reg_param)
        elif regularization == 'l2':
            regularizer = L2Regularization(reg_param)
        elif regularization is None:
            regularizer = None
        else:
            raise ValueError(
                'Regularizer {} is not defined'.format(regularization))

        # Select loss function
        if loss == 'hinge':
            self.loss = HingeLoss(regularizer)
        elif loss == 'squared':
            self.loss = SquaredLoss(regularizer)
        else:
            raise ValueError('Loss function {} is not defined'.format(loss))

        self.model = None



    def fit(self, features, targets, batch_size=None, max_iter=1000):
        """
        Fits a gradient descent learner to the features and targets. The
        pseudocode for the fitting algorithm is as follow:
          - Initialize the model parameters to uniform random values in the
            interval [-0.1, +0.1].
          - While not converged:
            - Compute the gradient of the loss with respect to the current
              batch.
            - Update the model parameters by moving them in the direction
              opposite to the current gradient. Use the learning rate as the
              step size.
        For the convergence criteria, compute the loss over all examples. If
        this loss changes by less than 1e-4 during an update, assume that the
        model has converged. If this convergence criteria has not been met
        after max_iter iterations, also assume convergence and terminate.

        You should include a bias term by APPENDING a column of 1s to your
        feature matrix. The bias term is then the last value in self.model.

        Arguments:
            features - (np.array) An Nxd array of features, where N is the
                number of examples and d is the number of features.
            targets - (np.array) A 1D array of targets of length N.
            batch_size - (int or None) The number of examples used in each
                iteration. If None, use all of the examples in each update.
            max_iter - (int) The maximum number of updates to perform.
        Modifies:
            self.model - (np.array) A 1D array of model parameters of length
                d+1. The +1 refers to the bias term.
        """
        feature_local = np.copy(features)
        f_shape = features.shape[0]
        f_shape1 = features.shape[1]
        f_array = np.ones((f_shape, 1))
        features = np.append(features,f_array, axis=1)
        
        self.model = np.random.uniform(low=-0.1, high=0.1, size=(features.shape[1],))

        

        loss_before = -np.inf
        loss_now = self.loss.forward(features, self.model, targets)


        count = 0
        tag = True
        while True:

            if count < max_iter:
                #print(count)
                if abs(loss_now - loss_before)>1e-4:
                    tag = True
                    #print(count)
                else:
                    break
            else:
                break
            

            if batch_size != None:
                #from stackoverflow
                sample = np.concatenate((features, np.vstack(targets)), 1)
                np.random.shuffle(sample)
                samp_abs = np.abs(sample)
                rand_feature = sample[:, :-1]
                rand_target = sample[:, -1]
                new_features = rand_feature[:batch_size, :]
                new_targets = rand_target[:batch_size]

                self.model -= self.learning_rate*self.loss.backward(new_features, self.model, new_targets)
                loss_before = loss_now
                loss_now = self.loss.forward(new_features, self.model, new_targets)

            else:
                self.model -= self.learning_rate * self.loss.backward(features, self.model, targets)
                loss_before = loss_now
                loss_now = self.loss.forward(features, self.model, targets)
            
            count += 1

    def predict(self, features):
        """
        Predicts the class labels of each example in features. Model output
        values at and above 0 are predicted to have label +1. Non-positive
        output values are predicted to have label -1.

        NOTE: your predict function should make use of your confidence
        function (see below).

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            predictions - (np.array) A 1D array of predictions of length N,
                where index d corresponds to the prediction of row N of
                features.
        """
        f_shape = features.shape[0]
        f_array = np.ones((f_shape, 1))
        p_array = np.append(features,f_array, axis=1)
        pred = []

        return np.array([-1 if np.dot(self.model, i, out=None) <= 0 else 1 for i in p_array])

    def confidence(self, features):
        """
        Returns the raw model output of the prediction. In other words, rather
        than predicting +1 for values above 0 and -1 for other values, this
        function returns the original, unquantized value.

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            confidence - (np.array) A 1D array of confidence values of length
                N, where index d corresponds to the confidence of row N of
                features.
        """
        f_shape = features.shape[0]
        f_array = np.ones((f_shape, 1))
        conf_array = np.append(features,f_array, axis=1)

        return np.array([np.dot(self.model, i, out=None) for i in conf_array])

