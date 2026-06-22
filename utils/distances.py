import numpy as np


def euclidean(x1: np.ndarray, x2: np.ndarray) -> float:
    """Euclidean (L2) distance between two vectors."""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan(x1: np.ndarray, x2: np.ndarray) -> float:
    """Manhattan (L1) distance between two vectors."""
    return np.sum(np.abs(x1 - x2))


def cosine_similarity(x1: np.ndarray, x2: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns value in [-1, 1]."""
    dot = np.dot(x1, x2)
    norm = np.linalg.norm(x1) * np.linalg.norm(x2)
    return dot / norm if norm > 0 else 0.0


def cosine_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Cosine distance = 1 - cosine_similarity."""
    return 1.0 - cosine_similarity(x1, x2)


def minkowski(x1: np.ndarray, x2: np.ndarray, p: int = 2) -> float:
    """Minkowski distance (generalized). p=1 is Manhattan, p=2 is Euclidean."""
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)