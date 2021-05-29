import numpy as np
from your_code import GradientDescent


class MultiClassGradientDescent:
    """
    Implements linear gradient descent for multiclass classification. Uses
    One-vs-All (OVA) classification for aggregating binary classification
    results to the multiclass setting.

    Arguments:
        loss - (string) The loss function to use. One of 'hinge' or 'squared'.
        regularization - (string or None) The type of regularization to use.
            One of 'l1', 'l2', or None. See regularization.py for more details.
        learning_rate - (float) The size of each gradient descent update step.
        reg_param - (float) The hyperparameter that controls the amount of
            regularization to perform. Must be non-negative.
    """

    def __init__(self, loss, regularization=None,
                 learning_rate=0.01, reg_param=0.05):
        self.loss = loss
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.reg_param = reg_param

        self.model = []
        self.classes = None

    def fit(self, features, targets, batch_size=None, max_iter=1000):
        """
        Fits a multiclass gradient descent learner to the features and targets
        by using One-vs-All classification. In other words, for each of the c
        output classes, train a GradientDescent classifier to determine whether
        each example does or does not belong to that class.

        Store your c GradientDescent classifiers in the list self.model. Index
        c of self.model should correspond to the binary classifier trained to
        predict whether examples do or do not belong to class c.

        Arguments:
            features - (np.array) An Nxd array of features, where N is the
                number of examples and d is the number of features.
            targets - (np.array) A 1D array of targets of size N. Contains c
                unique values (the possible class labels).
            batch_size - (int or None) The number of examples used in each
                iteration. If None, use all of the examples in each update.
            max_iter - (int) The maximum number of updates to perform.
        Modifies:
            self.model - (list) A list of c GradientDescent objects. The models
                trained to perform OVA classification for each class.
            self.classes - (np.array) A numpy array of the unique target
                values. Required to associate a model index with a target value
                in predict.
        """
        my_class = np.unique(targets)
        self.classes = my_class
        learning_rate = self.learning_rate
        reg_param = self.reg_param
        counter = 0
        for element in my_class:
            #for i in range(2):
            my_target = np.where(targets != element, -1, 1)
            my_target_size = np.size(my_target)
            target_abs = np.abs(my_target)
            my_gradient_descent = GradientDescent(self.loss, self.regularization, learning_rate, reg_param)
            my_gradient_descent.fit(features, my_target)
            self.model.append(my_gradient_descent)

        

    def predict(self, features):
        """
        Predicts the class labels of each example in features using OVA
        aggregation. In other words, predict as the output class the class that
        receives the highest confidence score from your c GradientDescent
        classifiers. Predictions should be in the form of integers that
        correspond to the index of the predicted class.

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            predictions - (np.array) A 1D array of predictions of length N,
                where index d corresponds to the prediction of row N of
                features.
        """
        f_shape = features.shape[0]
        c_size = np.size(self.classes)
        p_array = np.zeros((c_size, f_shape))

        for i in range(c_size):
            p_array[i, :] = self.model[i].confidence(features)

        p_max = np.argmax(p_array, axis=0)
        p_max_size = np.size(p_max)
        #print(np.array([self.classes[p_max[i]] for i in range(p_max_size)]))
        
        return np.array([self.classes[p_max[i]] for i in range(p_max_size)])
