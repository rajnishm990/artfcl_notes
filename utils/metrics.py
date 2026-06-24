import numpy as np 


def accuracy(y_true:np.ndarray , y_pred:np.ndarray):
    return np.mean(y_true==y_pred)

def precision(y_true : np.ndarray, y_pred: np.ndarray , positive_label=1) -> float:
    tp = np.sum((y_pred==positive_label) & (y_true == positive_label))
    fp = np.sum((y_pred!=positive_label) & (y_true == positive_label))
    return tp/(tp+fp) if (tp+fp) > 0 else 0.0 

def recall(y_true:np.ndarray , y_pred:np.ndarray , positive_label=1):
    tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
    fn = np.sum((y_pred != positive_label) & (y_true == positive_label))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0 

def f1_score(y_true:np.ndarray , y_pred: np.ndarray , positive_label=1)->float:
    p = precision(y_true , y_pred , positive_label)
    r = recall(y_true , y_pred , positive_label)
    return 2 * p * r / (p+r) if(p+r)>0 else 0.0 


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Returns a confusion matrix. Rows = actual, Cols = predicted."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for true, pred in zip(y_true, y_pred):
        cm[class_to_idx[true], class_to_idx[pred]] += 1
    return cm


# Regression Metrics 

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0