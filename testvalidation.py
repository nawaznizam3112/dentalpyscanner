import numpy as np
from dentscan.validation import sphere_report


def test_sphere_fit():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(2000, 3))
    pts = 12.7 * v / np.linalg.norm(v, axis=1, keepdims=True) + [1, 2, 3]
    r = sphere_report(pts, 25.4)
    assert abs(r["diameter_error_um"]) < 1
