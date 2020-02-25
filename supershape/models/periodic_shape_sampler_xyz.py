import torch
from torch import nn
from layers import primitive_wise_layers
from models import super_shape_sampler
import utils
from layers import layer_utils
from layers import super_shape_functions

EPS = 1e-7


class PeriodicShapeSamplerXYZ(super_shape_sampler.SuperShapeSampler):
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
        self.theta_dim = 2 if self.dim == 2 else 4
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

        self.encoder = nn.Sequential(
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      self.dim,
                                                      c64,
                                                      act=self.act),
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      c64,
                                                      c64 * 2,
                                                      act=self.act),
            primitive_wise_layers.PrimitiveWiseMaxPool(c64 * 2,
                                                       self.num_points))
        self.decoder = nn.Sequential(
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      c64 * 2 + self.theta_dim,
                                                      c64,
                                                      act=self.act),
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      c64,
                                                      c64,
                                                      act=self.act),
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      c64,
                                                      self.num_labels,
                                                      act='none'))

    def transform_circumference_angle_to_super_shape_world_cartesian_coord(
        self, thetas, radius, primitive_params, *args, points=None, **kwargs):
        assert not points is None

        assert len(thetas.shape) == 3, thetas.shape
        B, P, D = thetas.shape

        assert len(radius.shape) == 4, radius.shape
        # r = (B, n_primitives, P, dim - 1)
        r = radius.view(B, self.n_primitives, P, D)

        thetas_reshaped = thetas.view(B, 1, P, D)
        # B, n_primitives, P, dim
        cartesian_coord = super_shape_functions.polar2cartesian(
            r, thetas_reshaped)

        assert [*cartesian_coord.shape] == [B, self.n_primitives, P, self.dim]
        assert not torch.isnan(cartesian_coord).any()

        # (B, N, P)
        _, _, theta, phi = self.cartesian2polar(cartesian_coord,
                                                primitive_params, *args,
                                                **kwargs)
        primitive_r = ((cartesian_coord**2).sum(-1) + EPS).sqrt().unsqueeze(-1)

        periodic_net_r = self.get_periodic_net_r(thetas, radius, points)

        final_r = periodic_net_r + primitive_r
        if self.dim == 2:
            angles = theta.unsqueeze(-1)
        else:
            angles = torch.stack([theta, phi], axis=-1)

        #final_cartesian_coord = utils.sphere_polar2cartesian(
        #    final_r, angles.squeeze(2))
        final_cartesian_coord = utils.sphere_polar2cartesian(
            final_r,
            thetas.unsqueeze(1).repeat(1, self.n_primitives, 1, 1))

        if self.learn_pose:
            posed_cartesian_coord = self.project_primitive_to_world(
                final_cartesian_coord, primitive_params)
        else:
            posed_cartesian_coord = final_cartesian_coord

        # B, n_primitives, P, dim
        return posed_cartesian_coord

    def get_periodic_net_r(self, thetas, radius, points):
        # B, P, dim - 1
        assert len(thetas.shape) == 3
        assert thetas.shape[-1] == self.dim - 1
        assert points.shape[0] == thetas.shape[0]

        B, _, D = points.shape
        assert points.shape[1] == self.num_points

        _, P, Dn1 = thetas.shape
        N = 1
        # B, P, N
        thetas_transposed = thetas.view(B, 1, P,
                                        Dn1).transpose(1, 2).contiguous()

        # B, P, N, dim - 1
        sin = thetas_transposed.sin()
        cos = thetas_transposed.cos()

        # B, P, N, (dim - 1) * 2 = (2 or 4)
        sincos = torch.cat([sin, cos], axis=-1) * 100
        assert [*sincos.shape] == [B, P, N, (self.dim - 1) * 2], sincos.shape
        points_BxnPxNx2or3 = points.view(B, self.num_points, 1,
                                         self.dim).repeat(
                                             1, 1, self.n_primitives, 1)
        # Encoder expects B, num_points, N, D -> B, 1, N, self.encoder_dim
        encoded = self.encoder(points_BxnPxNx2or3)
        # sincos (B, P, 2 or 4)
        concated_BxP2xNx2or3pencoder_dim = torch.cat([
            encoded.view(B, 1, self.n_primitives, self.encoder_dim).repeat(
                1, P, 1, 1),
            sincos.repeat(1, 1, self.n_primitives // N, 1)
        ],
                                                     axis=-1)
        # decoder (B, P, self.n_primitives, 1)
        radius = self.decoder(concated_BxP2xNx2or3pencoder_dim).view(
            B, P, self.n_primitives, 1).transpose(1, 2).contiguous()
        radius = radius * self.last_scale

        return radius

    def get_sgn(self, *args, **kwargs):
        raise NotImplementedError
