"""
Fitzgibbon_et_al_numpy.py

Differentiable (symbolically identical) implementation of a simplified
direct least-squares ellipse fitting pipeline using NumPy only.
This mirrors the structure of the code below's PyTorch version.
(https://github.com/yehyunsuh/Acetabular-Cup-Pose-Estimator/blob/main/Fitzgibbon_et_al.py)

Reference:
    Fitzgibbon, A., Pilu, M., & Fisher, R. B. (1999).
    Direct least square fitting of ellipses.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 21(5), 476–480.

Author: Yehyun Suh
Date: 2025-09-04
"""

import numpy as np


def coords_euclidean_to_ep(X: np.ndarray) -> np.ndarray:
    """
    Map 2D coordinates to the extended parameter (EP) space for conic fitting.

    Args:
        X (np.ndarray): Input coordinates of shape [N, 2].

    Returns:
        np.ndarray: Transformed coordinates of shape [N, 6] with columns [x^2, xy, y^2, x, y, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape [N, 2].")
    x = X[:, 0]
    y = X[:, 1]
    A = x**2
    B = x * y
    C = y**2
    D = x
    E = y
    F = np.ones_like(x)
    return np.stack([A, B, C, D, E, F], axis=-1)  # [N, 6]


def coords_to_scatter_mat(X: np.ndarray) -> np.ndarray:
    """
    Compute the scatter matrix S = X_ep^T X_ep for least-squares ellipse fitting.

    Args:
        X (np.ndarray): Input coordinates of shape [N, 2].

    Returns:
        np.ndarray: Scatter matrix of shape [6, 6].
    """
    X_ep = coords_euclidean_to_ep(X)  # [N, 6]
    return X_ep.T @ X_ep  # [6, 6]


def _pinv_from_svd(M: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """
    Compute a numerically stable pseudo-inverse via SVD.

    Args:
        M (np.ndarray): Matrix to invert, shape [m, n].
        rcond (float): Cutoff for small singular values.

    Returns:
        np.ndarray: Pseudo-inverse of M with shape [n, m].
    """
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    # Invert with thresholding
    cutoff = rcond * S.max() if S.size else 0.0
    Sinv = np.where(S > cutoff, 1.0 / S, 0.0)
    return (Vt.T * Sinv) @ U.T


def fitzgibbon_ellipse(M: np.ndarray) -> np.ndarray:
    """
    Solve a simplified generalized eigenvalue problem proxy to find ellipse coefficients.

    NOTE:
    This mirrors the user's PyTorch routine: it takes the SVD-based inverse (pseudo-inverse)
    of the scatter matrix M, then takes the first left singular vector of M_inv as the
    conic parameters. This is *not* the full constrained Fitzgibbon method (which enforces
    the ellipse constraint), but preserves the user's original structure.

    Args:
        M (np.ndarray): 6x6 scatter matrix (XᵀX), float64.

    Returns:
        np.ndarray: Row vector of ellipse coefficients with shape [1, 6].
    """
    if M.shape != (6, 6):
        raise ValueError("Scatter matrix M must have shape [6, 6].")

    # Pseudo-inverse for numerical stability
    M_inv = _pinv_from_svd(M)

    # SVD of M_inv; first left-singular vector corresponds to max singular value direction.
    U, _, _ = np.linalg.svd(M_inv, full_matrices=False)

    conic_params = U[:, 0][None, :]  # shape [1, 6]
    return conic_params


def params_ep_to_ab(P: np.ndarray) -> np.ndarray:
    """
    Convert general conic parameters into ellipse parameters (center, axes, angle).

    Args:
        P (np.ndarray): Fitted conic parameters of shape [1, 6] (A, B, C, D, E, F).

    Returns:
        np.ndarray: Ellipse parameters [x_center, y_center, semi_major, semi_minor, angle_deg], shape [1, 5].
    """
    if P.shape != (1, 6):
        raise ValueError("P must have shape [1, 6].")

    A, B, C, D, E, F = [P[0, i] for i in range(6)]
    B_half = B / 2.0

    denom = 2.0 * (B_half**2 - A * C)
    if np.isclose(denom, 0.0):
        raise FloatingPointError("Degenerate conic (denominator ~ 0) while computing ellipse center.")

    x = (C * D - B_half * E) / denom  # center x
    y = (A * E - B_half * D) / denom  # center y

    mu_inv = (A * x**2 + 2.0 * B_half * x * y + C * y**2 - F)
    if np.isclose(mu_inv, 0.0):
        raise FloatingPointError("Degenerate conic (mu_inv ~ 0) while computing ellipse axes.")
    mu = 1.0 / mu_inv

    m11 = mu * A
    m12 = mu * B_half
    m22 = mu * C

    eig_sum = m11 + m22
    eig_diff = m11 - m22
    eig_cross = 2.0 * m12

    rad = np.sqrt(eig_diff**2 + eig_cross**2)
    lambda1 = 0.5 * (eig_sum + rad)
    lambda2 = 0.5 * (eig_sum - rad)

    # Guard against numerical negatives
    if lambda1 <= 0.0 or lambda2 <= 0.0:
        raise FloatingPointError("Non-positive quadratic form eigenvalues; not an ellipse.")

    a = 1.0 / np.sqrt(lambda1)  # semi-axes
    b = 1.0 / np.sqrt(lambda2)

    # Ensure a >= b (major >= minor)
    if a < b:
        a, b = b, a

    alpha = 0.5 * np.arctan2(-2.0 * B_half, C - A)  # radians
    alpha_deg = np.degrees(alpha)
    alpha_deg = (alpha_deg + 180.0) % 180.0

    return np.array([[x, y, a, b, alpha_deg]], dtype=np.float64)


def fitzgibbon_et_al(projected_landmarks: np.ndarray) -> np.ndarray:
    """
    Perform direct least-squares ellipse fitting for a given set of 2D landmarks.

    Args:
        projected_landmarks (np.ndarray): Landmark coordinates of shape [N, 3] (z ignored).

    Returns:
        np.ndarray: Fitted ellipse parameters as a vector:
            [center_x, center_y, semi_major, semi_minor, angle_deg], shape [5,].
    """
    projected_landmarks = np.asarray(projected_landmarks, dtype=np.float64)
    if projected_landmarks.ndim != 2 or projected_landmarks.shape[1] < 2:
        raise ValueError("projected_landmarks must have shape [N, >=2].")

    projected_landmarks_2D = projected_landmarks[:, :2]
    mean_2D = projected_landmarks_2D.mean(axis=0)  # [2]
    centered_2D = projected_landmarks_2D - mean_2D  # [N, 2]

    M = coords_to_scatter_mat(centered_2D)  # [6, 6]
    conic_params = fitzgibbon_ellipse(M)    # [1, 6]
    ellipse_params = params_ep_to_ab(conic_params)  # [1, 5]

    # Un-center the estimated ellipse center
    center_x = ellipse_params[0, 0] + mean_2D[0]
    center_y = ellipse_params[0, 1] + mean_2D[1]
    major_axis = ellipse_params[0, 2]
    minor_axis = ellipse_params[0, 3]
    angle_deg  = ellipse_params[0, 4]

    return np.array([center_x, center_y, major_axis, minor_axis, angle_deg], dtype=np.float64)