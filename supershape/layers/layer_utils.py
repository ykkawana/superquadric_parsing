import torch

EPS = 1e-7


def get_indicator(x, y, r):
    """get istropic indicator values.

    Args:
        x: x in catesian coordinate.
        y: y in catesian coordinate.
        r: radius in polar coordinate.

    Returns:
        indicator: indicator value. Positive in inside, negative in outside.
    """
    indicator = 1. - ((x**2. + y**2.) / (r**2.) + EPS).sqrt()
    return indicator


def get_m_periodic_sincos(theta, m, cycle_div=4.):
    sin = (theta * m / cycle_div).sin()
    cos = (theta * m / cycle_div).cos()
    return sin, cos
