import math
import torch
from models import model_utils


def test_deterministic_m_vector():
    """Test m_vector has max value at specified by m."""
    m = 5
    max_m = m + 1
    target_m_vector = torch.eye(max_m).view(1, 1, max_m, max_m)[:, :,
                                                                m, :].float()
    logits = target_m_vector * 10 + 1

    m_vector = model_utils.get_deterministic_m_vector(logits)

    torch.all(torch.eq(m_vector, target_m_vector))


def test_rotation_matrix():
    """Test if rotation matrix shape and values are as expected."""
    n = 2
    theta = 0.2
    rotation = torch.tensor([theta] * n).view(1, -1)
    target_rotation_matrix = torch.tensor(
        [math.cos(theta), -math.sin(theta),
         math.sin(theta),
         math.cos(theta)]).view(1, 1, 4).repeat(1, n, 1)
    rotation_matrix = model_utils.get_rotation_matrix(rotation)
    torch.all(torch.eq(target_rotation_matrix, rotation_matrix))
    assert [*target_rotation_matrix.shape] == [1, n, 4]

    # Test inverse rotation matrix
    inv_rotation_matrix = model_utils.get_rotation_matrix(rotation, inv=True)
    target_inv_rotation_matrix = torch.tensor(
        [math.cos(theta),
         math.sin(theta), -math.sin(theta),
         math.cos(theta)]).view(1, 1, 4).repeat(1, n, 1)
    torch.all(torch.eq(target_inv_rotation_matrix, inv_rotation_matrix))
