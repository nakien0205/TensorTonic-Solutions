import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    m, n = len([i for i in A]), len(A[0])
    array = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            array[i][j] = A[j][i]
    return array