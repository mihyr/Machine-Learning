import numpy as np

def confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the confusion matrix. The confusion 
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO 
    CLASSES (binary).
    
    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    
    #init elements of matrix
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    

    for i in range(len(actual)):
        
        if actual[i] == 0 and predictions[i] == 0:
            true_negatives += 1

        elif actual[i] == 0 and predictions[i] == 1:
            false_positives += 1

        elif actual[i] == 1 and predictions[i] == 0:
            false_negatives += 1

        elif actual[i] == 1 and predictions[i] == 1:
            true_positives += 1

    return np.array([[true_negatives, false_positives], [false_negatives, true_positives]])
     

def accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the accuracy:

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    c_matrix = confusion_matrix(actual, predictions)

    c_matrix_sum = np.sum(c_matrix)
    
    #Accuracy = sum of true positive and true negative, whole divided by sum of all elements of confusion matrix

    return (c_matrix[0, 0] + c_matrix[1, 1]) / c_matrix_sum
    

def precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    c_matrix = confusion_matrix(actual, predictions)
    true_positives = c_matrix[1, 1]
    false_positives = c_matrix[0, 1]
    false_negatives = c_matrix[1, 0]
    
    precision = 0
    recall = 0

    if (c_matrix[1, 1] + c_matrix[0, 1]):
        precision = c_matrix[1, 1]/(c_matrix[1, 1]+c_matrix[0, 1])

    if (c_matrix[1, 1] + c_matrix[1, 0]):
        recall = c_matrix[1, 1]/(c_matrix[1, 1]+c_matrix[1, 0])

    return precision, recall

def f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the F1-measure:

   https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Hint: implement and use the precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and 
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    precision_op, recall_op = precision_and_recall(actual, predictions)

    f1 = 0

    if (precision_op+ recall_op):
        f1 = 2*(precision_op*recall_op)/(precision_op+recall_op)

    return f1

#completed