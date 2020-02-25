import utils
import torch


def test_sphere_cartesian2polar():
    coord = torch.tensor([0., 0., 0.]).view(1, 3)
    r, angles = utils.sphere_cartesian2polar(coord)

    # Test if proper shape for 3d
    assert len(r.shape) == 2
    assert len(angles.shape) == 2
    assert angles.shape[-1] == 2

    # Test if nan occurs
    assert not torch.any(torch.isnan(r))
    assert not torch.any(torch.isnan(angles))

    coord = torch.tensor([
        0.,
        0.,
    ]).view(1, 2)
    r, angles = utils.sphere_cartesian2polar(coord)

    # Test if proper shape for 2d
    assert len(angles.shape) == 2
    assert angles.shape[-1] == 1

    coord = torch.tensor([1., 0., 0.])
    r, angles = utils.sphere_cartesian2polar(coord)

    # Check proper value
    assert r.item() == 1.
    assert angles[0].item() == 0.
    assert angles[1].item() == 0.
