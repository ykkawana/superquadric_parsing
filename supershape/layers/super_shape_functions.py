import torch
from layers import layer_utils
import utils

EPS = 1e-7


def rtheta2xy(r, thetas):
    xy = torch.stack([thetas.cos(), thetas.sin()], axis=-1) * r
    return xy


def rational_supershape_by_m(theta, m, n1, n2, n3, a, b):
    sin, cos = layer_utils.get_m_periodic_sincos(theta, m)
    r = rational_supershape(theta, sin, cos, n1, n2, n3, a, b)
    return r


def rational_supershape(theta, sin, cos, n1, n2, n3, a, b):
    def U(theta):
        u = (a * cos).abs()
        assert not torch.isnan(u).any(), (u, a)
        return u

    def V(theta):
        v = (b * sin).abs()
        assert not torch.isnan(v).any()
        return v

    def W(theta):
        w = (U(theta) /
             (n2 +
              (1. - n2) * U(theta) + EPS)) + (V(theta) /
                                              (n3 +
                                               (1. - n3) * V(theta) + EPS))
        assert not torch.isnan(w).any(), (n2, n3)
        return w

    r = (2.**(-(n1 + EPS))) * (n1 / (W(theta) + EPS) + 1. - n1)
    assert not torch.isnan(r).any(), n1
    return r


def implicit_rational_supershape_by_m(x, y, m, n1, n2, n3, a, b):
    theta = utils.safe_atan(y, x)
    sin, cos = layer_utils.get_m_periodic_sincos(theta, m)
    r = rational_supershape(theta, sin, cos, n1, n2, n3, a, b)
    indicator = layer_utils.get_indicator(x, y, r)
    return indicator


def implicit_rational_supershape(x, y, angles, n1, n2, n3, a, b):
    theta = utils.safe_atan(y, x) * angles
    assert not torch.isnan(theta).any(), (theta)
    r = rational_supershape(theta, theta.sin(), theta.cos(), n1, n2, n3, a, b)
    indicator = layer_utils.get_indicator(x, y, r)
    assert not torch.isnan(indicator).any(), indicator
    return indicator


def supershape(theta, sin, cos, n1, n2, n3, a, b):
    def U(theta):
        u = a * ((cos).abs()**n2)
        assert not torch.isnan(u).any(), (n2, a)
        return u

    def V(theta):
        v = b * ((sin).abs()**n3)
        assert not torch.isnan(v).any(), (n3, b)
        return v

    r = (U(theta) + V(theta) + EPS)**(-n1)
    assert not torch.isnan(r).any(), (r, n1)
    return r


def implicit_supershape_by_m(x, y, m, n1, n2, n3, a, b):
    theta = utils.safe_atan(y, x)
    sin, cos = layer_utils.get_m_periodic_sincos(theta, m)
    r = supershape(theta, sin, cos, n1, n2, n3, a, b)
    indicator = layer_utils.get_indicator(x, y, r)
    return indicator


def implicit_supershape(x, y, angles, n1, n2, n3, a, b):
    theta = utils.safe_atan(y, x) * angles
    r = supershape(theta, theta.sin(), theta.cos(), n1, n2, n3, a, b)
    indicator = layer_utils.get_indicator(x, y, r)
    return indicator
