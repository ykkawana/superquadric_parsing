import torch
from torch import nn
from layers import primitive_wise_layers
from models import super_shape_sampler
import utils
from layers import layer_utils


class PeriodicShapeSampler(super_shape_sampler.SuperShapeSampler):
    def __init__(self,
                 num_points,
                 *args,
                 last_bias=1.,
                 mode='scratch',
                 factor=1,
                 act='leaky',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = factor
        self.num_points = num_points
        self.num_labels = 1
        self.theta_dim = 2
        c64 = 64 // self.factor
        self.encoder_dim = c64 * 2
        #self.encoder_dim = c64 * 16
        #self.last_bias = .0001
        self.last_bias = last_bias
        if mode == 'scratch':
            self.last_scale = .1
        elif mode == 'delta':
            self.last_scale = .1
        self.act = act
        self.mode = mode  # one of ['scratch', 'delta']

        self.encoders = []
        self.decoders = []

        for idx in range(dim - 1):
            self.encoders.append(
                nn.Sequential(
                    primitive_wise_layers.PrimitiveWiseLinear(
                        self.n_primitives, self.dim, c64, act=self.act),
                    primitive_wise_layers.PrimitiveWiseLinear(
                        self.n_primitives, c64, c64 * 2, act=self.act),
                    primitive_wise_layers.PrimitiveWiseMaxPool(
                        c64 * 2, self.num_points)))
            self.decoders.append(
                nn.Sequential(
                    primitive_wise_layers.PrimitiveWiseLinear(
                        self.n_primitives, c64 * 2 + 2, c64, act=self.act),
                    primitive_wise_layers.PrimitiveWiseLinear(
                        self.n_primitives, c64, c64, act=self.act),
                    primitive_wise_layers.PrimitiveWiseLinear(
                        self.n_primitives, c64, self.num_labels, act='none')))

    def get_r(self, thetas, *args, points=None, idx=None, **kwargs):
        """Return radius.

        B, _, _, P, D = thetas.shape
        Args:
            thetas: (B, 1 or n_primitives, 1, P, D)
            points: (B, num_points, dim)
        Returns:
            radius: (B, n_primitives, max_m, P, D)
        """
        # B, 1 or N, 1, P, dim - 1
        assert len(thetas.shape) == 5
        assert points.shape[0] == thetas.shape[0]

        assert not points is None
        B, _, D = points.shape
        assert points.shape[1] == self.num_points

        B, N, _, P, Dn1 = thetas.shape
        # B, P, N
        thetas_transposed = thetas.view(B, N, P,
                                        Dn1).transpose(1, 2).contiguous()

        radiuses = []
        for idx2 in range(self.dim - 1):
            if not idx is None and not idx2 == idx:
                continue

            print('idx2', idx2)
            # If thetas last dim is one, then it's called from sgn. Thetas and
            # args are already sliced, only has one dim at the end.
            slice_idx = 0 if thetas.shape[-1] == 1 else idx2
            thetas_splitted = thetas_transposed[..., slice_idx].unsqueeze(-1)

            # B, P, N, 1
            sin = thetas_splitted.sin()
            cos = thetas_splitted.cos()

            # B, P, N, 2
            sincos = torch.cat([sin, cos], axis=-1) * 100
            assert [*sincos.shape] == [B, P, N, 2], sincos.shape
            points_BxnPxNx2or3 = points.view(B, self.num_points, 1,
                                             self.dim).repeat(
                                                 1, 1, self.n_primitives, 1)
            # Encoder expects B, num_points, N, D -> B, 1, N, self.encoder_dim
            encoded = self.encoders[idx2](points_BxnPxNx2or3)
            # sincos (B, P, 2 or 4)
            concated_BxP2xNx2or3pencoder_dim = torch.cat([
                encoded.view(B, 1, self.n_primitives, self.encoder_dim).repeat(
                    1, P, 1, 1),
                sincos.repeat(1, 1, self.n_primitives // N, 1)
            ],
                                                         axis=-1)
            # decoder (B, P, self.n_primitives, 1)
            radius = self.decoders[idx2](
                concated_BxP2xNx2or3pencoder_dim).view(
                    B, P, self.n_primitives, 1).transpose(1, 2).contiguous()
            radius = radius * self.last_scale
            if self.mode == 'scratch':
                radius = radius + self.last_bias
                radius = nn.functional.relu(radius) + 1e-7
                radius = radius.view(B, self.n_primitives, 1,
                                     P).repeat(1, 1, self.max_m, 1)
            elif self.mode == 'delta':
                # super.get_r (B, n_primitives, max_m, P, thetas' last dim)
                decoder_radius = radius.unsqueeze(2)
                new_args = []
                # For sin, cos
                for arg in args[:2]:
                    new_args.append(arg[..., slice_idx].unsqueeze(-1))
                # For n1, n2, n3, a, b
                for arg in args[2:]:
                    new_args.append(arg[..., slice_idx].unsqueeze(-1))
                super_r = super().get_r(thetas[..., slice_idx].unsqueeze(-1),
                                        *new_args, **kwargs)  #.unsqueeze(-1)
                radius = decoder_radius + super_r

                radius = nn.functional.relu(radius) + 1e-7
            else:
                raise NotImplementedError(
                    'Mode mush be one of scratch or delta')

            radiuses.append(radius)

        radius = torch.cat(radiuses, axis=-1)

        return radius
