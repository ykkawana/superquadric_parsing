from models import super_shape
import utils
import torch
from unittest import mock
import math

m = 2
n1 = 1
n2 = 1
n3 = 1
a = 1
b = 1
theta = math.pi / 2.


@mock.patch('models.model_utils.get_m_vector')
def test_get_primitive_params_m_vector_probabilistic(mocker):
    """Test if m vector is returned probabilistically or not.
    It's probabilistic in quadrics mode is false, eval mode, and train logits options is false. 
    """

    (theta_test_tensor, m_tensor, n1inv_tensor, n2_tensor, n3_tensor,
     ainv_tensor,
     binv_tensor) = utils.get_single_input_element(theta, m, n1, n2, n3, a, b)

    p = super_shape.SuperShapes(m, 1)
    p.eval()

    p.get_primitive_params()
    mocker.assert_called_with(p.logits, probabilistic=False)

    p.train()
    p.get_primitive_params()
    mocker.assert_called_with(p.logits, probabilistic=True)

    p_quadrics = super_shape.SuperShapes(4, 1, quadrics=True)

    p_quadrics.get_primitive_params()
    mocker.assert_called_with(p_quadrics.logits, probabilistic=False)

    p_quadrics.train()
    p_quadrics.get_primitive_params()
    mocker.assert_called_with(p_quadrics.logits, probabilistic=False)


def test_init():
    """Test tensor shapes."""
    # 2D
    super_shape.SuperShapes(m, 1, dim=2)
    # 3D
    super_shape.SuperShapes(m, 1, dim=3)
