from types import new_class
import numpy as np

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:
        https://en.wikipedia.org/wiki/K-means_clustering
        The KMeans algorithm has two steps:
        1. Update assignments
        2. Update the means
        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.
        Use only numpy to implement this algorithm.
        Args:
            n_clusters (int): Number of clusters to cluster the given data into.
        """
        self.assignments = None
        self.n_clusters = n_clusters
        self.means = None
        
                
    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.
        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        f_shape = np.shape(features)[0]
        self.assignments = np.empty(f_shape)
        num = self.n_clusters
        self.means = [features[np.random.permutation(np.arange(f_shape)), :] for x in range(num)]
        trigger = True

        while trigger==True:
            #create copy of assignments
            assignment_copy = np.copy(self.assignments)

            self.assignment_update(features)
            if np.ndarray.all(assignment_copy ==self.assignments):
                trigger = False
                break
            self.mean_update(features)

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.
        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        f_shape = np.shape(features)[0]
        pred_array = np.empty(f_shape)

        for i in features:
            dist = []
            for j in self.means:
                #print(i-j)
                norm_elem = np.linalg.norm(i - j, keepdims=False)
                dist.append(norm_elem)
                #np.append(dist,norm_elem)
                
            min_elem = np.min(dist)
            
            min_index_elem = dist.index(min_elem)
            assignment_index = np.where(features == i)[0]
            pred_array[assignment_index] = min_index_elem

        return pred_array

    def assignment_update(self,features):
        for i in features:
            dist = []
            
            for j in self.means:
                #print(i-j)
                norm_elem = np.linalg.norm(i - j, keepdims=False)
                dist.append(norm_elem)
                #np.append(dist,norm_elem)
                
            min_elem = np.min(dist)
            
            min_index_elem = dist.index(min_elem)
            assignment_index = np.where(features == i)[0]
            self.assignments[assignment_index] = min_index_elem


    def mean_update(self, features):
        assignment_stack = np.vstack(self.assignments)
        assignment_joined = np.concatenate((features, assignment_stack), axis=1)
        
        num = self.n_clusters
        for m in range(num*1):
            f_shape = np.shape(features)[1]
            self.means[m] = np.empty(f_shape)
            cluster = assignment_joined[assignment_joined[:, -1] == m]
            c_features = cluster[:, :-1]

            for n in range(f_shape):
                c_features_shape = np.shape(c_features)[0]
                c_features_sum = np.sum(c_features[:,n], axis=None)
                self.means[m][n] = c_features_sum/c_features_shape