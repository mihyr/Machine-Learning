import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def transform_data(features):
    """
    Data can be transformed before being put into a linear discriminator. If the data
    is not linearly separable, it can be transformed to a space where the data
    is linearly separable, allowing the perceptron algorithm to work on it. This
    function should implement such a transformation on a specific dataset (NOT 
    in general).

    Args:
        features (np.ndarray): input features
    Returns:
        transformed_features (np.ndarray): features after being transformed by the function
    """
    
    transformed_features = features

class Perceptron():
    def __init__(self, max_iterations=200):
        """
        This implements a linear perceptron for classification. A single
        layer perceptron is an algorithm for supervised learning of a binary
        classifier. The idea is to draw a linear line in the space that separates
        the points in the space into two partitions. Points on one side of the 
        line are one class and points on the other side are the other class.


        Args:
            max_iterations (int): the perceptron learning algorithm stops after 
            this many iterations if it has not converged.

        """
        self.max_iterations = max_iterations
        self.M = None
        self.N = None

    def fit(self, features, targets):
        """
        Fit a single layer perceptron to features to classify the targets, which
        are classes (-1 or 1). This function should terminate either after
        convergence (dividing line does not change between interations) or after
        max_iterations (defaults to 200) iterations are done. Here is pseudocode for 
        the perceptron learning algorithm:

        begin initialize weights
            while not converged or not exceeded max_iterations
                for each example in features
                    if example is misclassified using weights
                    then weights = weights + example * label_for_example
            return weights
        end
        
        Note that label_for_example is either -1 or 1.

        Use only numpy to implement this algorithm. 

        Args:
            features (np.ndarray): 2D array containing inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (saves model and training data internally)
        """
        f_shape = np.shape(features)
        self.N = np.zeros(f_shape[0])

        f_size = np.size(features)
        w_size = np.size(self.M)

        elem = np.ones((f_size))
        elem_shape = np.shape(elem)

        for i in range(elem_shape[0]):
            for j in range(1, elem_shape[1]):
                elem[i, j] = np.power(features[i],j)
        #from https://stackoverflow.com/questions/41186356/make-the-matrix-multiplication-operator-work-for-scalars-in-numpy
        self.M = np.linalg.inv(elem.T @ elem) @ elem.T @ targets

        for k in range(w_size):
            sorted_features = np.sort(features)
            feature_power_k =  np.power(sorted_features,k)
            self.N += self.M[k] * feature_power_k

        self.elem_shape = features 

    def predict(self, features):
        """
        Given features, a 2D numpy array, use the trained model to predict target 
        classes. Call this after calling fit.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        f_shape = np.shape(features)
        predict = np.zeros(f_shape[0])
        w_size = np.size(self.M)
        #print(f_shape)
        for i in range(w_size):
            feature_power_i =  np.power(features,i)
            predict += self.M[i]*feature_power_i
        #print(predict)
        return predict

    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the perceptron fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (plots to the active figure)
        """
        plt.figure()
        plt.plot(np.sort(self.elem_shape), self.N, 'g')  
        plt.scatter(features, targets)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.show()
        #plt.savefig('plot.png')
