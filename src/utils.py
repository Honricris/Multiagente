import numpy as np

def generate_shift(dimension, bounds):
    """ Calculate the 80% central range of the search space"""
    low, high = bounds
    center = (low + high) / 2
    full_range = high - low
    reduced_range = full_range * 0.8
    
    shift_low = center - (reduced_range / 2)
    shift_high = center + (reduced_range / 2)
    
    # Generate random shift vector within restricted boundaries
    return np.random.uniform(shift_low, shift_high, dimension)

def generate_rotation(dimension):
    """Produces an orthogonal matrix for coordinate rotation"""

    a = np.random.randn(dimension, dimension)
    q, r = np.linalg.qr(a)
    return q

def apply_transform(x, shift_vector, rotation_matrix):
    """Calculates z = M(x - o) to transform input space"""

    return np.dot(rotation_matrix, (x - shift_vector))