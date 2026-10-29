"""Accuracy validation against a known sphere."""
import numpy as np
from scipy.optimize import least_squares


def fit_sphere(points):
    p = np.asarray(points)
    c0 = p.mean(axis=0)
    r0 = np.linalg.norm(p - c0, axis=1).mean()

    def resid(x):
        return np.linalg.norm(p - x[:3], axis=1) - x[3]

    sol = least_squares(resid, np.r_[c0, r0])
    return sol.x[:3], sol.x[3], resid(sol.x)


def sphere_report(points, nominal_diameter_mm):
    _, r, res = fit_sphere(points)
    return {
        "fitted_diameter_mm": 2 * r,
        "diameter_error_um": (2 * r - nominal_diameter_mm) * 1000,
        "rms_um": float(np.sqrt((res ** 2).mean()) * 1000),
        "max_dev_um": float(np.abs(res).max() * 1000),
    }
