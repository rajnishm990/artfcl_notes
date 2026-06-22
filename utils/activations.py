import numpy as np


# SIGMOID

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid: 1 / (1 + exp(-z)). Clipped for numerical stability."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid: s(z) * (1 - s(z))."""
    s = sigmoid(z)
    return s * (1 - s)


# RElu

def relu(z: np.ndarray) -> np.ndarray:
    """ReLU: max(0, z)."""
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if z > 0 else 0."""
    return (z > 0).astype(float)


# Leaky ReLU 

def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU: z if z > 0 else alpha * z."""
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Derivative of Leaky ReLU."""
    return np.where(z > 0, 1.0, alpha)


# Tanh 

def tanh(z: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent."""
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of tanh: 1 - tanh(z)^2."""
    return 1 - np.tanh(z) ** 2


#  Softmax 

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax function. Numerically stable version.
    Works for 1D (single sample) or 2D (batch) inputs.
    """
    if z.ndim == 1:
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)
    else:
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)