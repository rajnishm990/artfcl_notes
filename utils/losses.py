import numpy as np


# Regression Losses 

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error loss."""
    return np.mean((y_true - y_pred) ** 2)


def mse_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Gradient of MSE with respect to y_pred."""
    n = len(y_true)
    return (2 / n) * (y_pred - y_true)


def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error loss."""
    return np.mean(np.abs(y_true - y_pred))


#  Classification Losses 

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Binary cross-entropy loss. y_pred should be probabilities in (0, 1)."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_gradient(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Gradient of binary cross-entropy with respect to y_pred."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    n = len(y_true)
    return (1 / n) * (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred))


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """
    Categorical cross-entropy loss.
    y_true: one-hot encoded (n_samples, n_classes)
    y_pred: softmax probabilities (n_samples, n_classes)
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Hinge loss for SVM. y_true should be {-1, +1}.
    L = max(0, 1 - y_true * y_pred)
    """
    return np.mean(np.maximum(0, 1 - y_true * y_pred))