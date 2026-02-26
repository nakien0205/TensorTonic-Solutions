import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    pe = np.zeros((seq_len, d_model))
    
    # 'pos' is a column vector of shape (seq_len, 1)
    pos = np.arange(seq_len)[:, np.newaxis]
    
    # 'i' represents the indices for the dimensions (0, 2, 4...)
    # We only need to compute this for half the d_model
    i = np.arange(0, d_model, 2)
    
    # Compute the denominator (div_term)
    # Using exp(log(...)) is often more numerically stable than power
    div_term = np.exp(i * -(np.log(base) / d_model))
    
    # Apply sin to even indices (0, 2, 4...)
    pe[:, 0::2] = np.sin(pos * div_term)
    
    # Apply cos to odd indices (1, 3, 5...)
    # If d_model is odd, we need to ensure the slice matches
    pe[:, 1::2] = np.cos(pos * div_term[:d_model//2])
    
    return pe