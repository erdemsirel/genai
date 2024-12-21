import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the cosine simalirity between two vectors.
    :param a: The first vector of shape (n_features,).
    :param b: The second vector of shape (n_features,).
    :return: The cosine simalirity between the two vectors as numpy scalar (shape ()).
    """
    assert len(a.shape) == 1, "a must be a vector"
    assert a.shape == b.shape, "a and b must have the same shape"

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
