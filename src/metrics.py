import numpy as np


def weighted_correlation(y_true, y_pred, weights):
    """Return the competition's weighted Pearson correlation.

    Non-finite rows are ignored. Degenerate inputs return ``nan`` instead of
    raising or producing an infinite score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(weights)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    weights = weights[mask]

    w_sum = np.sum(weights)
    if len(y_true) < 2 or w_sum <= 0:
        return np.nan

    y_true_mean = np.sum(weights * y_true) / w_sum
    y_pred_mean = np.sum(weights * y_pred) / w_sum

    y_true_centered = y_true - y_true_mean
    y_pred_centered = y_pred - y_pred_mean

    cov = np.sum(weights * y_true_centered * y_pred_centered)
    var_true = np.sum(weights * y_true_centered ** 2)
    var_pred = np.sum(weights * y_pred_centered ** 2)

    if var_true <= 0 or var_pred <= 0:
        return np.nan

    return cov / np.sqrt(var_true * var_pred)
