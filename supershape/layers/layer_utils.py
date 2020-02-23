import torch

EPS = 1e-7


def get_indicator(x, y, z, r1, r2, phi):
    """get istropic indicator values.

    Args:
        x: x in cartesian coordinate.
        y: y in cartesian coordinate.
        r1: radius in polar coordinate.
        r2: radius in polar coordinate. Defaults to 1.
        z: z in cartesian coordinate. Defaulst to 0.
        phi: polar coordinate of r2

    Returns:
        indicator: indicator value. Positive in inside, negative in outside.
    """
    numerator = (x**2. + y**2. + z**2.)
    denominator = ((phi.cos()**2) * (r1**2. - 1) + 1 + EPS)
    indicator = 1. - (1. / (r2 + EPS)) * (numerator / denominator + EPS).sqrt()
    return indicator


def get_m_periodic_sincos(theta, m, cycle_div=4.):
    sin = (theta * m / cycle_div).sin()
    cos = (theta * m / cycle_div).cos()
    return sin, cos
