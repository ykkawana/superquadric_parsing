from models import super_shape
from models import periodic_shape_sampler
import utils
import torch
from torch import nn
import math
from layers import super_shape_functions
from unittest import mock
import numpy as np

EPS = 1e-7


@mock.patch("models.super_shape_sampler.SuperShapeSampler.get_r")
def test_get_r_2d(mocker):
    batch = 3
    m = 4
    n = 2
    P = 7
    points_num = 5
    dim = 2

    sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                          m,
                                                          n,
                                                          mode='delta')
    mock_decoder = type(
        '', (nn.Module, ), {
            'forward':
            lambda self, x: torch.tensor([0.]).view(1, 1, 1, 1).repeat(
                batch, P, n, 1)
        })()
    sampler.decoders = [mock_decoder] * (dim - 1)
    mocker.return_value = torch.zeros([batch, n, m, P, dim - 1])

    batched_points = torch.zeros([batch, points_num, dim]).float()
    batched_thetas_N_1 = torch.zeros([batch, 1, 1, P, dim - 1]).float()

    r = sampler.get_r(batched_thetas_N_1, points=batched_points)

    assert [*r.shape] == [batch, n, m, P, dim - 1]

    batched_thetas_N_n = torch.zeros([batch, n, 1, P, dim - 1]).float()

    r = sampler.get_r(batched_thetas_N_n, points=batched_points)

    assert [*r.shape] == [batch, n, m, P, dim - 1]


@mock.patch("models.super_shape_sampler.SuperShapeSampler.get_r")
def test_get_r_3d(mocker):
    batch = 3
    m = 4
    n = 2
    P = 7
    points_num = 5
    dim = 3

    sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                          m,
                                                          n,
                                                          dim=dim,
                                                          mode='delta')
    mock_decoder = type(
        '', (nn.Module, ), {
            'forward':
            lambda self, x: torch.tensor([0.]).view(1, 1, 1, 1).repeat(
                batch, P, n, 1)
        })()
    sampler.decoder = [mock_decoder] * (dim - 1)
    mocker.return_value = torch.zeros([batch, n, m, P, 1])

    batched_points = torch.zeros([batch, points_num, dim]).float()
    batched_thetas_N_1 = torch.zeros([batch, 1, 1, P, dim - 1]).float()

    r = sampler.get_r(batched_thetas_N_1, points=batched_points)

    assert [*r.shape] == [batch, n, m, P, dim - 1]

    batched_thetas_N_n = torch.zeros([batch, n, 1, P, dim - 1]).float()

    r = sampler.get_r(batched_thetas_N_n, points=batched_points)

    assert [*r.shape] == [batch, n, m, P, dim - 1]


def test_transform_circumference_angle_to_super_shape_radius_2d():
    # If m = 4 and all other params is 1 except n, then the shape is square.
    # It's vertical and horizontal length is very close to one,
    # (0.5 and -0.5 in coordinate).
    batch = 3
    m = 4
    n = 2
    n1 = 1
    n2 = 1
    n3 = 1
    a = 1
    b = 1
    theta = math.pi / 2.
    P = 7
    points_num = 5
    dim = 2

    (theta_test_tensor, m_tensor, n1inv_tensor, n2_tensor, n3_tensor,
     ainv_tensor,
     binv_tensor) = utils.get_single_input_element(theta, m, n1, n2, n3, a, b)

    sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                          m,
                                                          n,
                                                          mode='delta')
    mock_decoder = type(
        '', (nn.Module, ), {
            'forward':
            lambda self, x: torch.tensor([0.]).view(1, 1, 1, 1).repeat(
                batch, P, n, 1)
        })()
    sampler.decoders = [mock_decoder] * (dim - 1)

    preset_params = utils.generate_multiple_primitive_params(
        m, n1, n2, n3, a, b)
    batched_theta_test_tensor = theta_test_tensor.repeat(batch).view(
        batch, 1, 1).repeat(1, P, dim - 1)
    batched_points = torch.tensor([0.,
                                   0.]).view(1, 1,
                                             2).repeat(batch, points_num, 1)
    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        batched_theta_test_tensor, preset_params, points=batched_points)

    assert [*radius.shape] == [batch, n, P, dim - 1]
    # All primitives length must be around 0.5.
    target_radius = torch.tensor([0.5] * P).view(1, 1, P, 1).repeat(
        batch, n, 1, dim - 1)
    assert torch.allclose(target_radius, radius)

    # Test in more complicated shape
    m = 6
    n1 = 3
    n2 = 10
    n3 = 5

    (theta_test_tensor, m_tensor, n1inv_tensor, n2_tensor, n3_tensor,
     ainv_tensor,
     binv_tensor) = utils.get_single_input_element(theta, m, n1, n2, n3, a, b)

    target_radius = super_shape_functions.rational_supershape_by_m(
        batched_theta_test_tensor, m_tensor, n1inv_tensor, n2_tensor,
        n3_tensor, ainv_tensor,
        binv_tensor).view(batch, 1, -1, dim - 1).repeat(1, n, 1, dim - 1)

    sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                          m,
                                                          n,
                                                          mode='delta')
    sampler.decoders = [mock_decoder] * (dim - 1)
    preset_params = utils.generate_multiple_primitive_params(
        m, n1, n2, n3, a, b)
    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        batched_theta_test_tensor, preset_params, points=batched_points)

    assert torch.allclose(target_radius, radius)


def test_transform_circumference_angle_to_super_shape_radius_3d():
    # If m = 4 and all other params is 1 except n, then the shape is square.
    # It's vertical and horizontal length is very close to one,
    # (0.5 and -0.5 in coordinate).
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

    rotations = [[0., 0., 0.]] * (dim - 1)
    transitions = [[0., 0., 0.]] * (dim - 1)
    linear_scales = [[1., 1., 1.]] * (dim - 1)

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

    (_, m_tensor, n1inv_tensor, n2_tensor, n3_tensor, ainv_tensor,
     binv_tensor) = utils.get_single_input_element(theta, m, n1, n2, n3, a, b)

    theta_test_points = [[0., 0.], [math.pi / 2., 0.],
                         [math.pi / 2., math.pi / 2.]]
    P = len(theta_test_points)
    batched_theta_test_tensor = torch.tensor(theta_test_points).view(
        1, 3, dim - 1).repeat(batch, 1, 1)

    sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                          m,
                                                          n,
                                                          dim=dim,
                                                          mode='delta')
    mock_decoder = type(
        '', (nn.Module, ), {
            'forward':
            lambda self, x: torch.tensor([0.]).view(1, 1, 1, 1).repeat(
                batch, P, n, 1)
        })()
    sampler.decoders = [mock_decoder] * (dim - 1)

    batched_points = torch.tensor([0.] * batch * points_num * dim).view(
        batch, points_num, dim)

    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        batched_theta_test_tensor, preset_params, points=batched_points)

    assert [*radius.shape] == [batch, n, P, dim - 1]

    # All primitives length must be around 0.5.
    target_radius = torch.tensor([0.5] * P)
    assert torch.allclose(
        target_radius.view(1, 1, P, 1).repeat(batch, n, 1, dim - 1),
        radius), radius.min()


def test_transform_circumference_angle_to_super_shape_world_cartesian_coord_2d(
):
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
    P = 7
    dim = 2
    points_num = 5

    rotations = [[0.], [math.pi / 2]]
    transitions = [[0., 0.], [1., 0.]]

    sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                          m,
                                                          n,
                                                          mode='delta',
                                                          dim=dim)
    mock_decoder = type(
        '', (nn.Module, ), {
            'forward':
            lambda self, x: torch.tensor([0.]).view(1, 1, 1, 1).repeat(
                batch, P, n, 1)
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

    batched_theta_test_tensor = torch.tensor(
        [theta] * P * (dim - 1)).repeat(batch).view(batch, P, dim - 1)

    batched_radius_tensor = torch.tensor([0.5] * P * (dim - 1)).view(
        1, 1, P, dim - 1).repeat(batch, n, 1, 1)
    points = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        batched_theta_test_tensor, batched_radius_tensor, preset_params)

    assert [*points.shape] == [batch, n, P, dim]
    target_points = torch.tensor([[0., 0.5],
                                  [0.5, 0.]]).view(1, n, -1,
                                                   dim).repeat(batch, 1, P, 1)
    assert torch.allclose(points, target_points, atol=EPS)


def test_transform_circumference_angle_to_super_shape_world_cartesian_coord_3d(
):
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

    rotations = [[0., 0., 0.], [0., math.pi / 2., 0.]]
    transitions = [[0., 0., 0.], [1., 0., 0.]]
    linear_scales = [[1., 1., 1.]] * n

    sampler = periodic_shape_sampler.PeriodicShapeSampler(points_num,
                                                          m,
                                                          n,
                                                          mode='delta',
                                                          dim=dim)
    mock_decoder = type(
        '', (nn.Module, ), {
            'forward':
            lambda self, x: torch.tensor([0.]).view(1, 1, 1, 1).repeat(
                batch, P, n, 1)
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
        transitions=transitions,
        linear_scales=linear_scales)

    theta_test_points = [[0., 0.], [math.pi / 2., 0.],
                         [math.pi / 2., math.pi / 2.]]
    batched_theta_test_tensor = torch.tensor(theta_test_points).view(
        1, 3, dim - 1).repeat(batch, 1, 1)
    P = len(theta_test_points)
    batched_radius_tensor = torch.tensor([0.5] * batch * n * P *
                                         (dim - 1)).view(batch, n, P, dim - 1)
    points = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        batched_theta_test_tensor, batched_radius_tensor, preset_params)

    assert [*points.shape] == [batch, n, P, dim]
    target_points = torch.tensor([[[0.25, 0., 0.], [0, 0.25, 0], [0., 0.,
                                                                  0.5]],
                                  [[1., 0., -0.25], [1, 0.25, 0.],
                                   [1.5, 0.,
                                    0.]]]).view(1, n, P,
                                                dim).repeat(batch, 1, 1, 1)
    print(points[0])
    print(target_points[0])
    assert torch.allclose(points, target_points, atol=EPS)


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
                batch, sample_num**dim, n, 1)
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
