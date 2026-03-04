import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.array(x)
    ndim = x.ndim
    if ndim == 2:
        axis = (0,)
    elif ndim == 4:
        axis = (0, 2, 3)
        gamma = np.array(gamma).reshape(1, -1, 1, 1)
        beta = np.array(beta).reshape(1, -1, 1, 1)

    u = np.mean(x, axis=axis, keepdims=True)
    o = np.var(x, axis=axis, keepdims=True)

    x_hat = (x - u) / np.sqrt(o + eps)
    y = gamma * x_hat + beta
    
    return y