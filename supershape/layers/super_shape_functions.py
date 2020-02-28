import torch
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

    #r = (2.**(-(n1 + EPS))) * (n1 / (W(theta) + EPS) + 1. - n1)
    r = 1. / (n1 * W(theta) + 1. - n1 + EPS)
    assert not torch.isnan(r).any(), n1
    return r


def supershape(theta, sin, cos, n1, n2, n3, a, b):
    def U(theta):
        u = (a * cos).abs()**n2
        assert not torch.isnan(u).any(), (n2, a)
        return u

    def V(theta):
        v = (b * sin).abs()**n3
        assert not torch.isnan(v).any(), (n3, b)
        return v

    r = (U(theta) + V(theta) + EPS)**(-n1)
    assert not torch.isnan(r).any(), (r, n1)
    return r
