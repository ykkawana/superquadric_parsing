import torch
from torch import nn
from layers import gumbel_softmax

EPS = 1e-7

def get_probabilistic_m_vector(logits, dim=2, hard=False):
    """Return weight vector for m coefficient probabilistically.
    Args:
        logits (1, n_primitives, max_m)
        dim: which column is for m coeffient
        hard (bool): return one hot vector if true.
    Return:
        selector  (1, n_primitives, max_m)
    """
    assert len(logits.shape) == 3
    selector = gumbel_softmax.gumbel_softmax(nn.functional.relu(logits + EPS),
                              dim=dim,
                              hard=hard)
    return selector


def get_deterministic_m_vector(logits, dim=2):
    """Return weight vector for m coefficient probabilistically.
    Args:
        logits (1, n_primitives, max_m)
        dim: which column is for m coeffient
    Return:
        selector  (1, n_primitives, max_m)
    """
    assert len(logits.shape) == 3
    y = nn.functional.relu(logits + 1e-7).argmax(axis=dim, keepdim=True)
    #y_onehot = torch.FloatTensor(1, self.n_primitives, self.max_m, 1)
    #y_onehot = y_onehot.zero_().to(y.get_device())
    y_onehot = torch.zeros_like(logits)

    y_onehot.scatter_(dim, y, 1)
    return y_onehot


def get_m_vector(logits, probabilistic=True):
    if probabilistic:
        m_vector = get_probabilistic_m_vector(logits)
    else:
        m_vector = get_deterministic_m_vector(logits)
    return m_vector


def get_rotation_matrix(rotation, inv=False):
    sgn = -1. if inv else 1.
    upper_rotation_matrix = torch.cat([rotation.cos(), -sgn * rotation.sin()],
                                      axis=-1)
    lower_rotation_matrix = torch.cat(
        [sgn * rotation.sin(), rotation.cos()], axis=-1)
    rotation_matrix = torch.stack(
        [upper_rotation_matrix, lower_rotation_matrix], axis=-2)
    return rotation_matrix


