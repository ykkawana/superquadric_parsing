import torch
import math
from collections import defaultdict


def safe_atan(y, x):
    theta = torch.atan2(y, x)
    theta = torch.where(theta >= 0, theta, (2 * math.pi + theta))
    return theta


def generate_grid_samples(grid_size,
                          batch=1,
                          sampling='grid',
                          sample_num=100,
                          device='cpu'):
    """
  Arguments:
    grid_size (int, list): sample from [-grid_size, grid_size] range if it's int, otherwise sample within grid_size if it's list
    batch (int): batch size. Sample size will be batch size * grid samples
    sampling (str): sample gridwise if it's `grid`, sample by uniform dist. if it's `uniform`.
    sample_num (int): number of samples when sampling is `uniform`
    device: if set, data will be initialized on that device. otherwise will be stored on `cpu`.
  Returns:
    xs (B, sample size)
    ys (B, sample size)
  """
    if isinstance(grid_size, list):
        sampling_start, sampling_end = grid_size
    else:
        sampling_start, sampling_end = -grid_size, grid_size
    if sampling == 'grid':
        sampling_range = torch.linspace(sampling_start,
                                        sampling_end,
                                        sample_num,
                                        device=device)
        xs, ys = torch.meshgrid([sampling_range, sampling_range])

        def contiguous_batch(s):
            return s.contiguous().view(1, -1).repeat(batch, 1)

        xs_batched = contiguous_batch(xs)
        ys_batched = contiguous_batch(ys)
    elif sampling == 'uniform':

        def sample_uniform():
            return torch.empty(sample_num * batch,
                               device=device).uniform_(sampling_start,
                                                       sampling_end)

        sampling_point_x = sample_uniform()
        sampling_point_y = sample_uniform()
        xs, ys = torch.meshgrid(sampling_point_x, sampling_point_y)

        def contiguous_batch(s):
            return s.contiguous().view(batch, -1)

        xs_batched = contiguous_batch(xs)
        ys_batched = contiguous_batch(ys)
    else:
        raise NotImplementedError('no such sampling mode')
    return xs_batched, ys_batched


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
                                     logit=None):
    n1 = torch.tensor([1. / n1]).float().view(1, 1)
    n2 = torch.tensor([n2]).float().view(1, 1)
    n3 = torch.tensor([n3]).float().view(1, 1)
    a = torch.tensor([1. / a]).float().view(1, 1)
    b = torch.tensor([1. / b]).float().view(1, 1)
    transition = torch.tensor(transition).float().view(1, 1, 2)
    rotation = torch.tensor(rotation).float().view(1, 1)
    linear_scale = torch.tensor(linear_scale).float().view(1, 1, 2)

    if logit:
        m_vector = linear_scale = torch.tensor(logit).float().view(1, 1, -1)
    else:
        logit = [0.] * (m + 1)
        logit[m] = 1.
        m_vector = torch.tensor(logit).float().view(1, 1, m + 1)

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
                                       rotations=[
                                           0.,
                                           0.,
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
            logit=None)
        for key in param:
            params[key].append(param[key])
    return_param = {}
    for key in params:
        if key == 'prob':
            continue
        return_param[key] = torch.cat(params[key], axis=1)
    return_param['prob'] = None

    return return_param
