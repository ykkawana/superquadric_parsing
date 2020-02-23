import torch
import math
from collections import defaultdict
from external.QuaterNet.common import quaternion
import numpy as np
import warnings


def safe_atan(y, x):
    theta = torch.atan2(y, x)
    theta = torch.where(theta >= 0, theta, (2 * math.pi + theta))
    return theta


def generate_grid_samples(grid_size,
                          batch=1,
                          sampling='grid',
                          sample_num=100,
                          device='cpu',
                          dim=2):
    """Generate mesh grid.
    Args:
        grid_size (int, list, dict):
            if type is int, sample between [-grid_size, grid_size] range for all
            dims.
            if type is list, sample between grid_size for all dims. 
            if type is dict, For each dim, sample between each element of `range` in grid_size 
            with number of samples specified by `sample_num` in grid_size.
        batch (int): batch size. Sample size will be batch size * grid samples
        sampling (str): sample gridwise if it's `grid`, sample by uniform dist. if it's `uniform`.
        sample_num (int): number of samples. If grid_type is dict, this argument will be ignored.
        device: if set, data will be initialized on that device. otherwise will be stored on `cpu`.
        dim: 2 or 3
    Returns:
        coord (B, sample size, dim)
    """
    assert dim in [2, 3]
    if isinstance(grid_size, list):
        sampling_start, sampling_end = grid_size
    elif isinstance(grid_size, int):
        sampling_start, sampling_end = -grid_size, grid_size
    else:
        warnings.warn('sample_num will be ignored')

    def contiguous_batch(s):
        return s.contiguous().view(1, -1).repeat(batch, 1)

    ranges = []
    for idx in range(dim):
        if isinstance(grid_size, dict):
            sampling_start, sampling_end = grid_size['range'][idx]
            sample_num = grid_size['sample_num'][idx]
        if sampling == 'grid':
            sampling_points = torch.linspace(sampling_start,
                                             sampling_end,
                                             sample_num,
                                             device=device)
        elif sampling == 'uniform':
            sampling_points = torch.empty(sample_num * batch,
                                          device=device).uniform_(
                                              sampling_start, sampling_end)
        else:
            raise NotImplementedError('no such sampling mode')

        ranges.append(sampling_points)

    coord_list = [contiguous_batch(s) for s in torch.meshgrid(ranges)]
    return torch.stack(coord_list, axis=-1)


def get_single_input_element(theta, m, n1, n2, n3, a, b):
    n1inv = 1. / n1
    ainv = 1. / a
    binv = 1. / b
    scatter_point_size = 5
    theta_test_tensor = torch.tensor([theta])
    m_tensor = torch.tensor([m])
    n1inv_tensor = torch.tensor([n1inv])
    n2_tensor = torch.tensor([n2])
    n3_tensor = torch.tensor([n3])
    ainv_tensor = torch.tensor([ainv])
    binv_tensor = torch.tensor([binv])
    return (theta_test_tensor, m_tensor, n1inv_tensor, n2_tensor, n3_tensor,
            ainv_tensor, binv_tensor)


def generate_single_primitive_params(m,
                                     n1,
                                     n2,
                                     n3,
                                     a,
                                     b,
                                     rotation=0.,
                                     transition=[
                                         0.,
                                         0.,
                                     ],
                                     linear_scale=[1., 1.],
                                     logit=None,
                                     dim=2):
    assert dim in [2, 3]
    n1 = torch.tensor([1. / n1]).float().view(1, 1, 1).repeat(1, 1, dim - 1)
    n2 = torch.tensor([n2]).float().view(1, 1, 1).repeat(1, 1, dim - 1)
    n3 = torch.tensor([n3]).float().view(1, 1, 1).repeat(1, 1, dim - 1)
    a = torch.tensor([1. / a]).float().view(1, 1, 1).repeat(1, 1, dim - 1)
    b = torch.tensor([1. / b]).float().view(1, 1, 1).repeat(1, 1, dim - 1)
    transition = torch.tensor(transition).float().view(1, 1, dim)
    if dim == 2:
        rotation = torch.tensor(rotation).float().view(1, 1, 1)
    else:
        rotation = torch.tensor(rotation).float().view(1, 1, 4)
    linear_scale = torch.tensor(linear_scale).float().view(1, 1, dim)

    if logit:
        m_vector = linear_scale = torch.tensor(logit).float().view(
            1, 1, -1, 1).repeat(1, 1, 1, dim - 1)
    else:
        logit = [0.] * (m + 1)
        logit[m] = 1.
        m_vector = torch.tensor(logit).float().view(1, 1, m + 1, 1).repeat(
            1, 1, 1, dim - 1)

    return {
        'n1': n1,
        'n2': n2,
        'n3': n3,
        'a': a,
        'b': b,
        'm_vector': m_vector,
        'rotation': rotation,
        'transition': transition,
        'linear_scale': linear_scale,
        'prob': None
    }


def generate_multiple_primitive_params(m,
                                       n1,
                                       n2,
                                       n3,
                                       a,
                                       b,
                                       rotations_angle=[
                                           [0.],
                                           [0.],
                                       ],
                                       transitions=[[
                                           0.,
                                           0.,
                                       ], [
                                           0.,
                                           0.,
                                       ]],
                                       linear_scales=[[1., 1.], [1., 1.]],
                                       logit=None):
    params = defaultdict(lambda: [])
    n = len(transitions)
    assert len(transitions) == len(rotations_angle), len(linear_scales)
    dim = len(transitions[0])
    assert dim in [2, 3]

    if dim == 2:
        assert len(rotations_angle[0]) == 1
        rotations = rotations_angle
    else:
        assert len(rotations_angle[0]) == 3
        rotations = convert_angles_to_quaternions(rotations_angle)

    for idx in range(n):
        param = generate_single_primitive_params(
            m,
            n1,
            n2,
            n3,
            a,
            b,
            rotation=rotations[idx],
            transition=transitions[idx],
            linear_scale=linear_scales[idx],
            logit=None,
            dim=dim)
        for key in param:
            params[key].append(param[key])
    return_param = {}
    for key in params:
        if key == 'prob':
            continue
        return_param[key] = torch.cat(params[key], axis=1)
    return_param['prob'] = None

    return return_param


def convert_angles_to_quaternions(rotations_angle):
    rotations = []
    for rotation in rotations_angle:
        rotations.append(
            quaternion.euler_to_quaternion(np.array(rotation), 'xyz').tolist())
    return rotations
