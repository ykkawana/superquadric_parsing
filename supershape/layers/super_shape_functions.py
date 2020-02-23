import torch
from layers import layer_utils
import utils

EPS = 1e-7


def polar2cartesian(radius, angles):
    """Convert polar coordinate to cartesian coordinate.
    Args:
        r: radius (B, N, P, 1 or 2) last dim is one in 2D mode, two in 3D.
        angles: angle (B, 1, P, 1 or 2) 
    """
    dim = radius.shape[-1]
    dim2 = angles.shape[-1]
    P = radius.shape[-2]
    P2 = angles.shape[-2]
    assert dim == dim2
    assert dim in [1, 2]
    assert P == P2

    theta = angles[..., 0]
    phi = torch.zeros([1], device=angles.device) if dim == 1 else angles[...,
                                                                         1]
    r1 = radius[..., 0]
    r2 = torch.ones([1], device=radius.device) if dim == 1 else radius[..., 1]

    phicosr2 = phi.cos() * r2
    cartesian_coord_list = [
        theta.cos() * r1 * phicosr2,
        theta.sin() * r1 * phicosr2
    ]

    # 3D
    if dim == 2:
        cartesian_coord_list.append(phi.sin() * r2)
    return torch.stack(cartesian_coord_list, axis=-1)


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


def implicit_rational_supershape(coord, angles, n1, n2, n3, a, b, m_vector):
    dim = coord.shape[-1]
    x = coord[..., 0]
    y = coord[..., 1]
    z = torch.zeros([1], device=coord.device) if dim == 2 else coord[..., 2]
    angles1 = angles[..., 0]
    angles2 = torch.ones([1], device=coord.device) if dim == 2 else angles[...,
                                                                           1]
    assert angles.shape[-1] == dim - 1
    theta = utils.safe_atan(y, x) * angles1
    assert not torch.isnan(theta).any(), (theta)
    r1_m = rational_supershape(theta, theta.sin(), theta.cos(), n1[..., 0],
                               n2[..., 0], n3[..., 0], a[..., 0],
                               b[..., 0]) * m_vector[..., 0]

    # B, n_primitives, P
    r1 = r1_m.sum(2)
    phi = utils.safe_atan(z * r1.unsqueeze(2) * x.cos(), x)
    phi_angled = phi * angles2
    assert not torch.isnan(phi).any(), (phi)
    r2 = torch.ones([1], device=phi.device).view(1, 1, 1, 1) if dim == 2 else (
        rational_supershape(phi_angled, phi_angled.sin(), phi_angled.cos(), n1[
            ..., 1], n2[..., 1], n3[..., 1], a[..., 1], b[..., 1]) *
        m_vector[..., 1]).sum(2).unsqueeze(2)

    indicator = layer_utils.get_indicator(x, y, z, r1.unsqueeze(2), r2,
                                          phi).squeeze(2)
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
