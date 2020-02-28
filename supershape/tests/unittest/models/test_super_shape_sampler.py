from models import super_shape
from models import super_shape_sampler
import utils
import torch
from torch import nn
import math
from layers import super_shape_functions
from external.QuaterNet.common import quaternion
import numpy as np

EPS = 1e-7


def test_reshape_params_check_shape_2d():
    m = 4
    n = 2
    n1 = 1
    n2 = 1
    n3 = 1
    a = 1
    b = 1

    dim = 2
    sampler = super_shape_sampler.SuperShapeSampler(m, n, dim=dim)
    preset_params = utils.generate_multiple_primitive_params(
        m, n1, n2, n3, a, b)


def test_reshape_params_check_shape_3d():
    m = 4
    n = 2
    n1 = 1
    n2 = 1
    n3 = 1
    a = 1
    b = 1

    dim = 3
    sampler = super_shape_sampler.SuperShapeSampler(m, n, dim=dim)
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
    dim = 2
    points_num = 5

    rotations = [[0.], [0.]]
    transitions = [[0., 0.], [1., 0.]]

    sampler = super_shape_sampler.SuperShapeSampler(m, n, dim=dim)
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

    cartesian_coord = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        thetas, radius, preset_params)

    assert [*cartesian_coord.shape] == [batch, n, P, dim]
    target_cartesian_coord_n1 = torch.tensor([[0., 1.], [1., 0.],
                                              [0.5, 0.5]]).view(1, P, dim)
    target_cartesian_coord_n2 = target_cartesian_coord_n1.clone()
    target_cartesian_coord_n2[..., 0] += 1
    target_cartesian_coord = torch.cat(
        [target_cartesian_coord_n1, target_cartesian_coord_n2], axis=0)

    assert torch.allclose(cartesian_coord[0], target_cartesian_coord, atol=EPS)


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

    rotations = [[0., 0., 0.], [0., 0., 0.]]
    transitions = [[0., 0., 0.], [1., 0., 0.]]
    linear_scales = [[1., 1., 1.]] * n

    sampler = super_shape_sampler.SuperShapeSampler(m, n, dim=dim)

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

    cartesian_coord = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        thetas, radius, preset_params)

    assert [*cartesian_coord.shape] == [batch, n, P, dim]
    target_cartesian_coord_n1 = torch.tensor([[0., 1., 0.], [1., 0., 0.],
                                              [0.5, 0.5, 0.], [-1., 0., 0.],
                                              [0., 0., 1.]]).view(1, P, dim)
    target_cartesian_coord_n2 = target_cartesian_coord_n1.clone()
    target_cartesian_coord_n2[..., 0] += 1
    target_cartesian_coord = torch.cat(
        [target_cartesian_coord_n1, target_cartesian_coord_n2], axis=0)

    assert torch.allclose(cartesian_coord[0],
                          target_cartesian_coord,
                          atol=EPS * 1e+1)


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
    dim = 2
    P = 7

    (theta_test_tensor, m_tensor, n1inv_tensor, n2_tensor, n3_tensor,
     ainv_tensor,
     binv_tensor) = utils.get_single_input_element(theta, m, n1, n2, n3, a, b)

    target_radius = torch.tensor([1.] * P)

    sampler = super_shape_sampler.SuperShapeSampler(m, n, dim=dim)
    preset_params = utils.generate_multiple_primitive_params(
        m, n1, n2, n3, a, b)
    batched_theta_test_tensor = theta_test_tensor.repeat(batch).view(
        batch, 1, 1).repeat(1, P, dim - 1)
    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        batched_theta_test_tensor, preset_params)

    assert [*radius.shape] == [batch, n, P, dim - 1]
    # All primitives length must be around 0.5.
    assert torch.allclose(
        target_radius.view(1, 1, P, 1).repeat(batch, n, 1, dim - 1), radius)


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
    sampler = super_shape_sampler.SuperShapeSampler(m, n, dim=dim)

    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        batched_theta_test_tensor, preset_params)

    assert [*radius.shape] == [batch, n, P, dim - 1]

    # All primitives length must be around 0.5.
    target_radius = torch.tensor([1.] * P)
    assert torch.allclose(
        target_radius.view(1, 1, P, 1).repeat(batch, n, 1, dim - 1),
        radius), radius.min()


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

    rotations = [[0.], [math.pi / 2]]
    transitions = [[0., 0.], [0.5, 0.]]

    sampler = super_shape_sampler.SuperShapeSampler(m, n, dim=dim)
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
    sgn = sampler.transform_world_cartesian_coord_to_tsd(coord, preset_params)

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

        # Check corners
        assert torch.allclose(xmax,
                              torch.tensor([1.]) + transitions[idx][0],
                              atol=area_error_tol)
        assert torch.allclose(xmin,
                              -torch.tensor([1.]) + transitions[idx][0],
                              atol=area_error_tol)
        assert torch.allclose(ymax,
                              torch.tensor([1.]) + transitions[idx][1],
                              atol=area_error_tol)
        assert torch.allclose(ymin,
                              -torch.tensor([1.]) + transitions[idx][1],
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

    area_error_tol = 1e-1
    dim = 3
    device = 'cpu'

    grid_ranges = [
        {
            'range': [[-1.2, 2.2], [-1.2, 1.2], [0, 0]],
            'sample_num': [sample_num, sample_num, 1]
        },
        {  # for y-z plane
            'range': [[0, 0], [-1.2, 1.2], [-1.2, 1.2]],
            'sample_num': [1, sample_num, sample_num]
        }
    ]
    target = [[{
        'xmin': -1.0,
        'xmax': 1.0,
        'ymin': -1.0,
        'ymax': 1.0,
        'zmin': 0.,
        'zmax': 0.
    }, {
        'xmin': 0.0,
        'xmax': 2.0,
        'ymin': -1.0,
        'ymax': 1.0,
        'zmin': 0.,
        'zmax': 0.
    }],
              [{
                  'xmin': 0.,
                  'xmax': 0.,
                  'ymin': -1.0,
                  'ymax': 1.0,
                  'zmin': -1.0,
                  'zmax': 1.0
              }, {
                  'xmin': 0.,
                  'xmax': 0.,
                  'ymin': -1.0,
                  'ymax': 1.0,
                  'zmin': -1.0,
                  'zmax': 1.0
              }]]
    rotationss = [[[0., 0., 0.], [0., math.pi / 2, 0.]]] * len(grid_ranges)
    translationss = [[[0., 0., 0.], [1.0, 0., 0.]], [[0., 0., 0.],
                                                     [0., 0., 0.]]]
    linear_scaless = [[[1., 1., 1.], [1., 1., 1.]]] * len(grid_ranges)
    for idx in range(len(grid_ranges)):
        rotations = rotationss[idx]
        transitions = translationss[idx]
        linear_scales = linear_scaless[idx]

        sampler = super_shape_sampler.SuperShapeSampler(m, n, dim=dim)
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
        print(coord.shape)
        sgn = sampler.transform_world_cartesian_coord_to_tsd(
            coord, preset_params)

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


def test_extract_surface_point_std():
    batch = 1
    m = 4
    n = 2
    n1 = 1
    n2 = 1
    n3 = 1
    a = 1
    b = 1
    dim = 2

    rotations = [[0.], [0.]]
    transitions = [[0., 0.], [.5, 0.]]

    sampler = super_shape_sampler.SuperShapeSampler(m, n, dim=dim)
    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions)

    target_points = [[[0., 1.], [1., 0.]], [[0., 0.], [1., 1.]]]

    batched_target_points = torch.tensor(target_points).view(
        batch, n, len(target_points[0]), dim)

    print(batched_target_points)

    sgn = sampler.extract_surface_point_std(batched_target_points,
                                            preset_params)
    sgn_tanhed = nn.functional.tanh(sgn * 100)

    target_sgn = [[[0., 0., 1., -1], [-1., 1., 1., -1.]]]

    batched_target_sgn = torch.tensor(target_sgn).view(
        batch, n, n * len(target_points[0]))

    assert torch.allclose(batched_target_sgn, sgn_tanhed, atol=1e-4)


def test_forward():
    batch = 1
    m = 4
    n = 5
    n1 = 1
    n2 = 1
    n3 = 1
    a = 1
    b = 1
    dim = 3

    rotations = [[0., 0., 0.]] * n
    transitions = [[0., 0., 0.]] * n
    linear_scales = [[1., 1., 1.]] * n

    sampler = super_shape_sampler.SuperShapeSampler(m, n, dim=dim)
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

    grid_range = {
        'range': [[-math.pi, math.pi], [-math.pi / 2, math.pi / 2]],
        'sample_num': [5, 5]
    }
    batched_theta_test_tensor = utils.generate_grid_samples(grid_range,
                                                            batch=batch,
                                                            dim=dim - 1)

    point_grid_range = 5
    batched_points_test_tensor = utils.generate_grid_samples(point_grid_range,
                                                             batch=batch,
                                                             dim=dim)

    sampler.forward(preset_params,
                    coord=batched_points_test_tensor,
                    thetas=batched_theta_test_tensor)


def test_transform_world_cartesian_coord_to_tsd_2d_with_random_points():
    batch = 3
    m = 3
    n = 1
    n1 = 1
    n2 = 10
    n3 = 3
    a = 1
    b = 1
    theta = math.pi / 2.
    sample_num = 200

    dim = 2

    rotations = [[0.]]
    transitions = [[0., 0.]]
    linear_scales = [[1., 1.]]

    sampler = super_shape_sampler.SuperShapeSampler(m, n, dim=dim)
    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions,
        linear_scales=linear_scales,
        nn=n)
    """
    thetas_list = [math.pi / 2., 0., math.pi / 4.]
    P = len(thetas_list)
    batched_theta_test_tensor = torch.tensor(thetas_list).view(1, P,
                                                               dim - 1).repeat(
                                                                   batch, 1, 1)
    """

    batched_theta_test_tensor = utils.sample_spherical_angles(
        sample_num=sample_num, batch=batch, dim=dim)
    # B, N, P
    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        batched_theta_test_tensor, preset_params)
    # B, P, dim
    coord = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        batched_theta_test_tensor, radius, preset_params).view(batch, -1, dim)

    sgn = sampler.transform_world_cartesian_coord_to_tsd(coord, preset_params)
    assert torch.allclose(sgn, torch.zeros_like(sgn),
                          atol=1e-5), (sgn.min(), sgn.max())


def test_transform_world_cartesian_coord_to_tsd_3d_with_random_points():
    batch = 3
    m = 3
    n = 1
    n1 = 1
    n2 = 10
    n3 = 3
    a = 1
    b = 1
    theta = math.pi / 2.
    sample_num = 200
    points_num = 5
    P = 10

    dim = 3

    rotations = [[0., 0., 0.]]
    transitions = [[0., 0., 0.]]
    linear_scales = [[1., 1., 1.]]

    sampler = super_shape_sampler.SuperShapeSampler(m, n, dim=dim)
    preset_params = utils.generate_multiple_primitive_params(
        m,
        n1,
        n2,
        n3,
        a,
        b,
        rotations_angle=rotations,
        transitions=transitions,
        linear_scales=linear_scales,
        nn=n)

    batched_theta_test_tensor = utils.sample_spherical_angles(
        batch=batch, sgn_convertible=True, dim=dim)

    # B, N, P
    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        batched_theta_test_tensor, preset_params)
    # B, P, dim
    coord = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        batched_theta_test_tensor, radius, preset_params).view(batch, -1, dim)

    sgn = sampler.transform_world_cartesian_coord_to_tsd(coord, preset_params)
    assert torch.allclose(sgn, torch.zeros_like(sgn),
                          atol=1e-5), (sgn.max(), sgn.min())
