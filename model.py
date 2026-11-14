"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np

def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    X = np.asarray(X, dtype=float, copy=True)
    means = np.nanmean(X, axis=0)
    means = np.nan_to_num(means, nan=0.0)
    nan_mask = np.isnan(X)
    return np.where(nan_mask, means, X)

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    IQR = q3 - q1

    lower = q1 - k * IQR
    upper = q3 + k * IQR

    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    X = np.asarray(X, dtype=float)
    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    if numerator.shape[0] != denominator.shape[0]:
        raise ValueError("numerator and denominator must have the same length")

    return numerator / (denominator + eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    X1 = np.asarray(X, dtype=float)
    X1 = np.column_stack([X1, col])
    return X1

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    labels = np.asarray(labels)
    sorted_unique = np.unique(labels)
    return (labels[:, None] == sorted_unique[None, :]).astype(float)

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (mean, std)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    X = np.asarray(X)
    mean = np.asarray(mean)
    std = np.asarray(std)

    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    N = X.shape[0]
    Xb = np.hstack([np.ones((N, 1)), X])
    return Xb

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    samples = np.arange(n_samples)
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(samples)
    return shuffled

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (train_idx, val_idx, test_idx)

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    X = np.asarray(X)
    y = np.asarray(y)

    indices = np.asarray(indices, dtype=int)

    return X[indices], y[indices]

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    X = np.asarray(X, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)

    return X @ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(np.mean(np.abs(y_true - y_pred)))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0.0:
        return 0.0

    return 1 - ss_res/ss_tot

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residual = y_true - y_pred

    mean = float(np.mean(residual))
    std = float(np.std(residual))
    median = float(np.median(np.abs(residual)))

    return {
        "mean": mean,
        "std": std,
        "median_abs": median
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X = np.asarray(X, dtype=float)

    X_imp = impute_nan_with_mean(X)
    lower, uppper = compute_iqr_bounds(X_imp, iqr_k)
    return clip_columns(X_imp, lower, uppper)

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    X_num = np.asarray(X_num, dtype=float)

    numerator = X_num[:, ratio_num_idx]
    denominator = X_num[:, ratio_den_idx]

    ratio = make_ratio_feature(numerator, denominator)
    X_ext = append_column(X_num, ratio)

    if cat_labels is not None:
        cat_block = one_hot_encode(cat_labels)
        X_ext = np.hstack([X_ext, cat_block])

    return X_ext

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n = X.shape[0]

    indices = make_shuffled_indices(n, seed)
    train_idx, val_idx, test_idx = partition_indices(indices, train_ratio, val_ratio)

    X_train, y_train = subset_xy(X, y, train_idx)
    X_val, y_val = subset_xy(X, y, val_idx)
    X_test, y_test = subset_xy(X, y, test_idx)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    # TODO: Fit standardizer on train, transform all splits, prepend bias...
    mean, std = fit_standardizer(splits["X_train"])

    std_splits = {
        "X_train": add_bias_column(apply_standardizer(splits["X_train"], mean, std)),
        "X_val": add_bias_column(apply_standardizer(splits["X_val"], mean, std)),
        "X_test": add_bias_column(apply_standardizer(splits["X_test"], mean, std)),
        # Targets are left unchanged
        "y_train": splits["y_train"],
        "y_val": splits["y_val"],
        "y_test": splits["y_test"],
    }

    return std_splits, mean, std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        "residual_summary": residual_summary(y_true, y_pred),
    }

# Step 24 - house_price_pipeline
def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    # TODO: Run full clean->featurize->split->standardize->OLS->evaluate pipeline...
    X_clean = prepare_cleaned_features(X, iqr_k)
    X_feat = assemble_feature_matrix(X_clean, ratio_num_idx, ratio_den_idx, cat_labels)

    splits = make_train_val_test(X_feat, y, train_ratio, val_ratio, seed)
    std_splits, mean, std = standardize_and_add_bias(splits)

    theta = ols_fit(std_splits["X_train"], std_splits["y_train"])

    y_val_pred = ols_predict(std_splits["X_val"], theta)
    y_test_pred = ols_predict(std_splits["X_test"], theta)

    return {
        "theta": theta,
        "y_test": std_splits["y_test"],
        "y_test_pred": y_test_pred,
        "test_metrics": evaluate_predictions(std_splits["y_test"], y_test_pred),
        "val_metrics": evaluate_predictions(std_splits["y_val"], y_val_pred),
    }

