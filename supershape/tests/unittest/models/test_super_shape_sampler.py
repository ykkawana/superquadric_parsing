from models import super_shape
from models import super_shape_sampler
import utils
import torch
from torch import nn
import math
from layers import super_shape_functions
from external.QuaterNet.common import quaternion

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
    rotations = [[0., 0., 0., 0.]] * (dim - 1)
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

    target_radius = torch.tensor([0.5] * P)

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

    # Test in more complicated shape
    m = 6
    n1 = 3
    n2 = 10
    n3 = 5

    (theta_test_tensor, m_tensor, n1inv_tensor, n2_tensor, n3_tensor,
     ainv_tensor,
     binv_tensor) = utils.get_single_input_element(theta, m, n1, n2, n3, a, b)

    batched_theta_test_tensor = theta_test_tensor.repeat(batch).view(
        batch, 1, 1).repeat(1, P, dim - 1)

    target_radius = super_shape_functions.rational_supershape_by_m(
        batched_theta_test_tensor, m_tensor, n1inv_tensor, n2_tensor,
        n3_tensor, ainv_tensor, binv_tensor)

    sampler = super_shape_sampler.SuperShapeSampler(m, n, dim=dim)
    preset_params = utils.generate_multiple_primitive_params(
        m, n1, n2, n3, a, b)
    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        batched_theta_test_tensor, preset_params)

    assert torch.all(
        torch.eq(
            target_radius.view(batch, 1, P, 1).repeat(1, n, 1, dim - 1),
            radius))


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
    P = 7

    rotations = [[0., 0., 0., 0.]] * (dim - 1)
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

    (theta_test_tensor, m_tensor, n1inv_tensor, n2_tensor, n3_tensor,
     ainv_tensor,
     binv_tensor) = utils.get_single_input_element(theta, m, n1, n2, n3, a, b)
    batched_theta_test_tensor = theta_test_tensor.repeat(batch).view(
        batch, 1, 1).repeat(1, P, dim - 1)

    sampler = super_shape_sampler.SuperShapeSampler(m, n, dim=dim)

    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        batched_theta_test_tensor, preset_params)

    assert [*radius.shape] == [batch, n, P, dim - 1]

    # All primitives length must be around 0.5.
    target_radius = torch.tensor([0.5] * P)
    assert torch.allclose(
        target_radius.view(1, 1, P, 1).repeat(batch, n, 1, dim - 1), radius)

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
        n3_tensor, ainv_tensor, binv_tensor)

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
    radius = sampler.transform_circumference_angle_to_super_shape_radius(
        batched_theta_test_tensor, preset_params)

    assert torch.all(
        torch.eq(
            target_radius.view(batch, n, P, 1).repeat(1, 1, 1, dim - 1),
            radius))


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

    rotations = [[0.], [math.pi / 2]]
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

    batched_theta_test_tensor = torch.tensor([theta] * P).repeat(batch).view(
        batch, P)
    batched_radius_tensor = torch.tensor([0.5] * P).view(1, 1, P).repeat(
        batch, n, 1)
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
    P = 7
    dim = 3

    rotations = [[0., 0., 0.], [0., math.pi / 2., math.pi / 2.]]
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

    batched_theta_test_tensor = torch.tensor(
        [theta] * batch * P * (dim - 1)).view(batch, P, dim - 1)
    batched_radius_tensor = torch.tensor([0.5] * batch * n * P *
                                         (dim - 1)).view(batch, n, P, dim - 1)
    points = sampler.transform_circumference_angle_to_super_shape_world_cartesian_coord(
        batched_theta_test_tensor, batched_radius_tensor, preset_params)

    assert [*points.shape] == [batch, n, P, dim]
    target_points = torch.tensor([[0., 0., 0.5],
                                  [1.5, 0.,
                                   0.]]).view(1, n, -1,
                                              dim).repeat(batch, 1, P, 1)
    assert torch.allclose(points, target_points, atol=EPS)


def test_transform_world_cartesian_coord_to_tsd_2d():
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

    batched_theta_test_tensor = torch.tensor([theta]).repeat(batch).view(
        batch, 1, dim - 1)
    coord = utils.generate_grid_samples(grid_range,
                                        batch=batch,
                                        sample_num=sample_num,
                                        dim=dim)
    sgn = sampler.transform_world_cartesian_coord_to_tsd(coord, preset_params)

    sgn_bool = (sgn >= 0).float()
    assert [*sgn_bool.shape] == [batch, n, sample_num**2]

    for idx in range(n):
        sgn_p = sgn_bool[0, idx, :]
        x = xs[0][torch.where(sgn_p)]
        y = ys[0][torch.where(sgn_p)]
        xmax = x.max()
        ymax = y.max()
        xmin = x.min()
        ymin = y.min()

        width = xmax - xmin
        height = ymax - ymin

        assert torch.allclose(width * height,
                              torch.ones([1]),
                              atol=area_error_tol)
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
