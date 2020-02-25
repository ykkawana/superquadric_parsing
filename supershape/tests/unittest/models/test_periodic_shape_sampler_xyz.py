from models import super_shape
from models import periodic_shape_sampler_xyz
from models import super_shape_sampler
import utils
import torch
from torch import nn
import math
from layers import super_shape_functions
from unittest import mock
import numpy as np

EPS = 1e-7


@mock.patch(
    'models.periodic_shape_sampler_xyz.PeriodicShapeSamplerXYZ.get_periodic_net_r'
)
def test_transform_circumference_angle_to_super_shape_world_cartesian_coord_2d(
    mocker):
    """Test if posed points are in correct place."""
    batch = 3
    m = 4
    n = 2
    n1 = 1
    n2 = 1
    n3 = 1
    a = 1
    b = 1
    theta = math.pi / 2.
    dim = 2
    points_num = 5

    rotations = [[0.], [0.]]
    transitions = [[0., 0.], [1., 0.]]

    sampler = periodic_shape_sampler_xyz.PeriodicShapeSamplerXYZ(points_num,
                                                                 m,
                                                                 n,
                                                                 mode='delta',
                                                                 dim=dim)
    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions)

    thetas_list = [math.pi / 2., 0., math.pi / 4.]
    P = len(thetas_list)
    thetas = torch.tensor(thetas_list).view(1, P, dim - 1).repeat(batch, 1, 1)

    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        thetas, preset_params)

    batched_points = torch.tensor([0.] * batch * points_num * dim).view(
        batch, points_num, dim)
    margin = 0.5
    mocker.return_value = margin
    cartesian_coord = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        thetas, radius, preset_params, points=batched_points)

    assert [*cartesian_coord.shape] == [batch, n, P, dim]
    target_cartesian_coord_n1 = torch.tensor(
        [[0., 1. + margin], [1. + margin, 0.],
         [0.5 + math.sqrt(margin**2 / 2),
          0.5 + math.sqrt(margin**2 / 2)]]).view(1, P, dim)
    target_cartesian_coord_n2 = target_cartesian_coord_n1.clone()
    target_cartesian_coord_n2[..., 0] += 1
    target_cartesian_coord = torch.cat(
        [target_cartesian_coord_n1, target_cartesian_coord_n2], axis=0)

    print(cartesian_coord[0])
    print(target_cartesian_coord)
    assert torch.allclose(cartesian_coord[0], target_cartesian_coord, atol=EPS)


@mock.patch(
    'models.periodic_shape_sampler_xyz.PeriodicShapeSamplerXYZ.get_periodic_net_r'
)
def test_transform_circumference_angle_to_super_shape_world_cartesian_coord_3d(
    mocker):
    """Test if posed points are in correct place."""
    batch = 3
    m = 4
    n = 2
    n1 = 1
    n2 = 1
    n3 = 1
    a = 1
    b = 1
    theta = math.pi / 2.
    dim = 3
    points_num = 5

    rotations = [[0., 0., 0.], [0., 0., 0.]]
    transitions = [[0., 0., 0.], [1., 0., 0.]]
    linear_scales = [[1., 1., 1.]] * n

    sampler = periodic_shape_sampler_xyz.PeriodicShapeSamplerXYZ(points_num,
                                                                 m,
                                                                 n,
                                                                 mode='delta',
                                                                 dim=dim,
                                                                 rational=True)
    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions,
        linear_scales=linear_scales)

    thetas_list = [[math.pi / 2., 0.], [0., 0.], [math.pi / 4., 0.],
                   [math.pi, 0.], [math.pi / 2., math.pi / 2.]]
    P = len(thetas_list)
    thetas = torch.tensor(thetas_list).view(1, P, dim - 1).repeat(batch, 1, 1)

    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        thetas, preset_params)

    batched_points = torch.tensor([1.] * batch * points_num * dim).view(
        batch, points_num, dim)
    margin = 0.5
    mocker.return_value = margin
    cartesian_coord = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        thetas, radius, preset_params, points=batched_points)

    assert [*cartesian_coord.shape] == [batch, n, P, dim]
    target_cartesian_coord_n1 = torch.tensor(
        [[0., 1. + margin, 0.], [1. + margin, 0., 0.],
         [0.5 + math.sqrt(margin**2 / 2), 0.5 + math.sqrt(margin**2 / 2), 0.],
         [-1. - margin, 0., 0.], [0., 0., 1. + margin]]).view(1, P, dim)
    target_cartesian_coord_n2 = target_cartesian_coord_n1.clone()
    target_cartesian_coord_n2[..., 0] += 1
    target_cartesian_coord = torch.cat(
        [target_cartesian_coord_n1, target_cartesian_coord_n2], axis=0)

    print(cartesian_coord[0])
    print(target_cartesian_coord)
    assert torch.allclose(cartesian_coord[0],
                          target_cartesian_coord,
                          atol=EPS * 1e+1)


def test_transform_world_cartesian_coord_to_tsd_2d():
    """Test if area with positive signs is as expected size for all primitives 
    and each corner of shape (square) is as expected. 
    """
    batch = 3
    m = 4
    n = 2
    n1 = 1
    n2 = 1
    n3 = 1
    a = 1
    b = 1
    theta = math.pi / 2.
    sample_num = 200
    grid_range = [-1, 1.5]
    area_error_tol = 1e-1
    dim = 2
    points_num = 5

    rotations = [[0.], [math.pi / 2]]
    transitions = [[0., 0.], [0.5, 0.]]

    sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                          m,
                                                          n,
                                                          mode='delta',
                                                          dim=dim)
    mock_decoder = type(
        '', (nn.Module, ), {
            'forward':
            lambda self, x: torch.tensor([0.]).view(1, 1, 1, 1).repeat(
                batch, sample_num**dim, n, dim - 1)
        })()
    sampler.decoders = [mock_decoder] * (dim - 1)

    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions)

    batched_theta_test_tensor = torch.tensor([theta] * batch * (dim - 1)).view(
        batch, 1, dim - 1)
    coord = utils.generate_grid_samples(grid_range,
                                        batch=batch,
                                        sample_num=sample_num,
                                        dim=dim)

    batched_points = torch.tensor([0.] * batch * points_num * dim).view(
        batch, points_num, dim)
    sgn = sampler.transform_world_cartesian_coord_to_tsd(coord,
                                                         preset_params,
                                                         points=batched_points)

    sgn_bool = (sgn >= 0).float()
    assert [*sgn_bool.shape] == [batch, n, sample_num**dim]

    for idx in range(n):
        sgn_p = sgn_bool[0, idx, :]
        x = coord[0, :, 0][torch.where(sgn_p)]
        y = coord[0, :, 1][torch.where(sgn_p)]
        xmax = x.max()
        ymax = y.max()
        xmin = x.min()
        ymin = y.min()

        width = xmax - xmin
        height = ymax - ymin

        # Check area
        assert torch.allclose(width * height,
                              torch.ones([1]),
                              atol=area_error_tol)
        # Check corners
        assert torch.allclose(xmax,
                              torch.tensor([0.5]) + transitions[idx][0],
                              atol=area_error_tol)
        assert torch.allclose(xmin,
                              -torch.tensor([0.5]) + transitions[idx][0],
                              atol=area_error_tol)
        assert torch.allclose(ymax,
                              torch.tensor([0.5]) + transitions[idx][1],
                              atol=area_error_tol)
        assert torch.allclose(ymin,
                              -torch.tensor([0.5]) + transitions[idx][1],
                              atol=area_error_tol)


def test_transform_world_cartesian_coord_to_tsd_3d():
    batch = 3
    m = 4
    n = 2
    n1 = 1
    n2 = 1
    n3 = 1
    a = 1
    b = 1
    theta = math.pi / 2.
    sample_num = 200
    points_num = 5

    area_error_tol = 1e-1
    dim = 3
    device = 'cpu'

    # x-y plane, xmin is -0.25, xmax is 1.5, ymin is -0.25, ymax is 0.25
    grid_ranges = [
        {
            'range': [[-0.3, 1.7], [-0.3, 0.3], [0, 0]],
            'sample_num': [sample_num, sample_num, 1]
        },
        {  # for y-z plane
            'range': [[0, 0], [-0.3, 0.3], [-0.7, 0.7]],
            'sample_num': [1, sample_num, sample_num]
        }
    ]
    target = [[{
        'xmin': -0.25,
        'xmax': 0.25,
        'ymin': -0.25,
        'ymax': 0.25,
        'zmin': 0.,
        'zmax': 0.
    }, {
        'xmin': 0.5,
        'xmax': 1.5,
        'ymin': -0.25,
        'ymax': 0.25,
        'zmin': 0.,
        'zmax': 0.
    }],
              [{
                  'xmin': 0.,
                  'xmax': 0.,
                  'ymin': -0.25,
                  'ymax': 0.25,
                  'zmin': -0.5,
                  'zmax': 0.5
              }, {
                  'xmin': 0.,
                  'xmax': 0.,
                  'ymin': -0.25,
                  'ymax': 0.25,
                  'zmin': -0.25,
                  'zmax': 0.25
              }]]
    rotationss = [[[0., 0., 0.], [0., math.pi / 2, 0.]]] * len(grid_ranges)
    translationss = [[[0., 0., 0.], [1.0, 0., 0.]], [[0., 0., 0.],
                                                     [0., 0., 0.]]]
    linear_scaless = [[[1., 1., 1.], [1., 1., 1.]]] * len(grid_ranges)
    for idx in range(len(grid_ranges)):
        rotations = rotationss[idx]
        transitions = translationss[idx]
        linear_scales = linear_scaless[idx]

        sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                              m,
                                                              n,
                                                              mode='delta',
                                                              dim=dim)
        mock_decoder = type(
            '', (nn.Module, ), {
                'forward':
                lambda self, x: torch.tensor([0.]).view(1, 1, 1, 1).repeat(
                    batch, sample_num**(dim - 1), n, 1)
            })()
        sampler.decoders = [mock_decoder] * (dim - 1)

        sampler.to(device)
        preset_params = utils.generate_multiple_primitive_params(
            m,
            n1,
            n2,
            n3,
            a,
            b,
            rotations_angle=rotations,
            transitions=transitions,
            linear_scales=linear_scales)

        for param in preset_params:
            if not preset_params[param] is None:
                preset_params[param] = preset_params[param].to(device)

        grid_range = grid_ranges[idx]
        coord = utils.generate_grid_samples(grid_range,
                                            batch=batch,
                                            dim=dim,
                                            device=device)
        batched_points = torch.tensor([0.] * batch * points_num * dim).view(
            batch, points_num, dim)
        sgn = sampler.transform_world_cartesian_coord_to_tsd(
            coord, preset_params, points=batched_points)

        sgn_bool = (sgn >= 0).float()
        assert [*sgn_bool.shape
                ] == [batch, n, np.prod(grid_range['sample_num'])]

        for idx2 in range(n):
            sgn_p = sgn_bool[0, idx2, :]
            x = coord[0, :, 0][torch.where(sgn_p)]
            y = coord[0, :, 1][torch.where(sgn_p)]
            z = coord[0, :, 2][torch.where(sgn_p)]
            xmax = x.max()
            ymax = y.max()
            zmax = z.max()
            xmin = x.min()
            ymin = y.min()
            zmin = z.min()

            target_coord = target[idx][idx2]
            target_coord = {
                key: torch.tensor(target_coord[key])
                for key in target_coord
            }

            assert torch.allclose(xmax,
                                  target_coord['xmax'],
                                  atol=area_error_tol)
            assert torch.allclose(xmin,
                                  target_coord['xmin'],
                                  atol=area_error_tol)
            assert torch.allclose(ymax,
                                  target_coord['ymax'],
                                  atol=area_error_tol)
            assert torch.allclose(ymin,
                                  target_coord['ymin'],
                                  atol=area_error_tol)
            assert torch.allclose(zmax,
                                  target_coord['zmax'],
                                  atol=area_error_tol)
            assert torch.allclose(zmin,
                                  target_coord['zmin'],
                                  atol=area_error_tol)
