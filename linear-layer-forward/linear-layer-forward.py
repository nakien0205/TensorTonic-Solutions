def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    import numpy as np
    a = np.matmul(X, W) + np.array(b)
    return a.tolist()