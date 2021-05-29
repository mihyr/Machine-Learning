import numpy as np
import math
class Node():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Node classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Node classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)
        
        """
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def _fit_helper_fn(self, features, targets, tree_node, attribute_names):

        node_op = None
        ones = 1
        target_zeros = np.count_nonzero(targets == 0)
        target_ones = np.count_nonzero(targets == 1)

        if features.size == 0 and targets.size == 0:
            node_op =  Node(attribute_name='leaf')

        elif len(attribute_names) == 0 and target_ones > target_zeros:
            node_op =  Node(attribute_name='leaf', value=1)

        elif len(attribute_names) == 0 and target_ones<= target_zeros:
            node_op =  Node(attribute_name='leaf', value=0)

        elif target_ones == len(targets):
            node_op =  Node(attribute_name='leaf', value=1)

        elif target_zeros == len(targets):
            node_op =  Node(attribute_name='leaf', value=0)

        else:
            targetstack = np.vstack(targets)
            data_set = np.concatenate((features, targetstack), 1)
            gain={}

            for i in range(len(attribute_names)):
                gain[attribute_names[i]] = information_gain(features, i, targets)

            tree_node.attribute_name = max(gain, key=gain.get)
            tree_node.attribute_index = self.attribute_names.index(tree_node.attribute_name)
            attribute_index1 = attribute_names.index(tree_node.attribute_name)
            
            rc_data_set = data_set[data_set[:, attribute_index1] == 1]
            rc_features = rc_data_set[:,:-1]
            rc_features = np.delete(rc_features, attribute_index1, 1)
            rc_targets = rc_data_set[:, -1]
            lc_data_set = data_set[data_set[:, attribute_index1] == 0] 
            lc_features = lc_data_set[:,:-1]
            lc_features = np.delete(lc_features, attribute_index1, 1)
            lc_targets = lc_data_set[:, -1]
            
            new_attributes = attribute_names.copy()
            new_attributes.remove(tree_node.attribute_name)
            L_child = self._fit_helper_fn(lc_features, lc_targets, Node(), new_attributes)
            R_child = self._fit_helper_fn(rc_features, rc_targets, Node(), new_attributes)

            tree_node.branches.append(L_child)
            tree_node.branches.append(R_child)

            node_op =  tree_node
        return node_op

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        Output:
            VOID: It should update self.tree with a built decision tree.
        """
        self._check_input(features)

        update_tree = self._fit_helper_fn(features, targets, Node(), self.attribute_names)
        self.tree = update_tree

    def _prediction_helper_fn(self, point, tree_node):
        predict_op = None

        if tree_node.branches == []:
            predict_op= tree_node.value

        elif (point[tree_node.attribute_index] == 0):
            predict_op= self._prediction_helper_fn(point, tree_node.branches[0])

        else:
            predict_op= self._prediction_helper_fn(point, tree_node.branches[1])

        return predict_op

    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        Outputs:
            predictions (np.array): numpy array of size N array which has the predicitons 
            for the input data.
        """
        self._check_input(features)

        predict = np.zeros((features.shape[0]))

        index = 0
        for point in features:
            predict[index] = self._prediction_helper_fn(point, self.tree)
            index += 1

        return predict

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """
    #find unique attribute and target labels
    attribute_labels = np.unique(features[:, attribute_index])
    target_labels = np.unique(targets)

    #init dicts
    weight_p = {}
    weight_c = {}

    #init entropy attribute and set
    entropy_a = 0
    entropy_s = 0
    

    for q in target_labels:
        weight_pp = np.count_nonzero(targets == q)/targets.size
        entropy_s = entropy_s - weight_pp*math.log(weight_pp, 2)

    #iterate through m labels in attribute_labels
    for m in attribute_labels:
        attr_target_index = 0
        weight_p[m] = {}
        weight_c_nonzero = np.count_nonzero(features[:, attribute_index] == m) 
        weight_c[m] = weight_c_nonzero / features[:, attribute_index].size
        

        attr_index = np.argwhere(features[:, attribute_index] == m)
        attr_target = np.zeros((len(attr_index)))
        
        
        for n in attr_index:
            attr_target[attr_target_index] = targets[n[0]]   
            attr_target_index+= 1

            for o in target_labels:
                weight_p_nonzero_target =  np.count_nonzero(attr_target == o) 
                weight_p_nonzero_feature =  np.count_nonzero(features[:, attribute_index] == m)
                weight_p[m][o] =weight_p_nonzero_target / weight_p_nonzero_feature

    for m in weight_p.keys():
        entropy_l = 0
        for n in weight_p[m].values():
            entropy_l -= 0 if n == 0 else n * math.log(n, 2)
        entropy_a += weight_c[m] * entropy_l

    return entropy_s - entropy_a

if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Node(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Node(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
