import torch
from torch import nn
import math

from models import model_utils


class SuperShapes(nn.Module):
    def __init__(self,
                 max_m,
                 n_primitives,
                 rational=True,
                 debug=False,
                 train_logits=True,
                 train_linear_scale=True,
                 quadrics=False,
                 train_ab=True):
        super(SuperShapes, self).__init__()
        self.debug = debug
        self.n_primitives = n_primitives
        self.max_m = max_m + 1
        self.rational = rational
        self.quadrics = quadrics
        self.train_logits = train_logits
        self.train_ab = train_ab
        if self.rational:
            self.n23scale = 10.
            self.n23bias = 1.
        else:
            self.n23scale = 1.
            self.n23bias = 0.

        if self.quadrics:
            self.n23scale = 1.
            self.n23bias = 1e-7

        if self.quadrics:
            assert max_m >= 4, 'super quadrics musth have m bigger than 4.'

        # 1, n_primitives, max_m, 1
        if self.quadrics:
            self.logits = nn.Parameter(
                torch.eye(self.max_m).view(
                    1, 1, self.max_m, self.max_m)[:, :, 4, :].repeat(
                        1, n_primitives, 1).float() * 10)
        elif not self.train_logits:
            self.logits = nn.Parameter(
                torch.eye(self.max_m).view(1, 1, self.max_m, self.max_m)
                [:, :,
                 (self.max_m - 1), :].repeat(1, n_primitives, 1).float() * 10)
        else:
            self.logits = nn.Parameter(
                torch.Tensor(1, n_primitives, self.max_m))
        assert [*self.logits.shape] == [1, n_primitives, self.max_m]

        # Shape params
        # B=1, n_primitives, 1, 1
        self.n1 = nn.Parameter(torch.Tensor(1, self.n_primitives))
        self.n2 = nn.Parameter(torch.Tensor(1, self.n_primitives))
        self.n3 = nn.Parameter(torch.Tensor(1, self.n_primitives))

        # Pose params
        # B=1, n_primitives, 2
        self.transition = nn.Parameter(torch.Tensor(1, n_primitives, 2))
        # B=1, n_primitives, 1
        self.rotation = nn.Parameter(torch.Tensor(1, n_primitives))
        # B=1, n_primitives, 2
        self.linear_scale = nn.Parameter(torch.Tensor(1, n_primitives, 2))
        self.linear_scale.requires_grad = train_linear_scale

        # B=1, n_primitives, 1, 1
        self.a = nn.Parameter(torch.Tensor(1, self.n_primitives))
        self.b = nn.Parameter(torch.Tensor(1, self.n_primitives))
        assert [*self.a.shape] == [1, self.n_primitives]
        if not self.train_ab:
            self.a.requires_grad = False
            self.b.requires_grad = True

        # B=1, n_primitives
        self.prob = nn.Parameter(torch.Tensor(1, n_primitives))

        if self.quadrics:
            self.n1.requires_grad = False
            self.logits.requires_grad = False
        elif not self.train_logits:
            self.logits.requires_grad = False

        self.weight_init()

    def weight_init(self):
        if self.quadrics:
            torch.nn.init.ones_(self.n1)
            torch.nn.init.uniform_(self.n2, 0, 1)
            torch.nn.init.uniform_(self.n3, 0, 1)
        elif not self.train_logits:
            torch.nn.init.uniform_(self.n1, 0, 1)
            torch.nn.init.uniform_(self.n2, 0, 1)
            torch.nn.init.uniform_(self.n3, 0, 1)
        else:
            torch.nn.init.uniform_(self.n1, 0, 1)
            torch.nn.init.uniform_(self.n2, 0, 1)
            torch.nn.init.uniform_(self.n3, 0, 1)
            torch.nn.init.uniform_(self.logits, 0, 1)
        if self.train_ab:
            torch.nn.init.uniform_(self.a, 0.9, 1.1)
            torch.nn.init.uniform_(self.b, 0.9, 1.1)
        else:
            torch.nn.init.ones_(self.a)
            torch.nn.init.ones_(self.b)
        torch.nn.init.uniform_(self.rotation, 0, 1)
        torch.nn.init.uniform_(self.linear_scale, 0, 1)
        torch.nn.init.uniform_(self.transition, -1, 1)
        torch.nn.init.uniform_(self.prob, 0, 1)

    def get_primitive_params(self):
        probabilistic = self.training and self.train_logits and not self.quadrics
        m_vector = model_utils.get_m_vector(self.logits,
                                            probabilistic=probabilistic)
        linear_scale = nn.functional.tanh(self.linear_scale) + 1.1

        a = nn.functional.relu(self.a)
        b = nn.functional.relu(self.b)

        n1 = nn.functional.relu(self.n1)
        n2 = nn.functional.relu(self.n2 * self.n23scale) + self.n23bias
        n3 = nn.functional.relu(self.n3 * self.n23scale) + self.n23bias

        rotation = nn.functional.tanh(self.rotation) * math.pi

        prob = nn.functional.sigmoid(self.prob)

        return {
            'n1': n1,
            'n2': n2,
            'n3': n3,
            'a': a,
            'b': b,
            'm_vector': m_vector,
            'rotation': rotation,
            'transition': self.transition,
            'linear_scale': linear_scale,
            'prob': prob
        }

    def forward(self):
        return self.get_primitive_params()

    def get_m(self):
        m = torch.argmax(self.logits, axis=2)
        return m
