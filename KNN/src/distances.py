import numpy as np

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    #find length of m and n matrix
    row_M = X.shape[0]
    row_N = Y.shape[0]

    #init output_matrix
    output_matrix = np.empty(shape = (row_M, row_N))
    
    for m in range(row_M):
        for n in range(row_N):
           #find difference between elements of both rows -> square it -> sum it -> sqrt it 
            element_dist = np.sqrt(np.sum(np.square(X[m] - Y[n])))
            output_matrix[m,n] = element_dist

    return output_matrix


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    row_M = X.shape[0]
    row_N = Y.shape[0]

    #init output_matrix
    output_matrix = np.empty(shape = (row_M, row_N))
    
    for m in range(row_M):
        for n in range(row_N):
           #find difference between elements of both rows -> sum it
            element_dist = np.sum(abs(X[m] - Y[n]))
            output_matrix[m,n] = element_dist

    return output_matrix