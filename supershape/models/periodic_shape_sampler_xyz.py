import torch
from torch import nn
from layers import primitive_wise_layers
from models import super_shape_sampler
import utils
from layers import super_shape_functions
from models import point_net

EPS = 1e-7


class PeriodicShapeSamplerXYZ(super_shape_sampler.SuperShapeSampler):
    def __init__(self,
                 num_points,
                 *args,
                 last_bias=1.,
                 last_scale=.1,
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
        self.last_scale = last_scale

        self.act = act

        self.encoder = point_net.PointNet(self.encoder_dim,
                                          dim=self.dim,
                                          factor=self.factor,
                                          act=self.act)
        self.decoder = nn.Sequential(
            primitive_wise_layers.PrimitiveWiseLinear(self.n_primitives,
                                                      self.encoder_dim +
                                                      self.theta_dim + 1,
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
        thetas_reshaped = thetas.view(B, 1, P, D)

        assert len(radius.shape) == 4, radius.shape
        # r = (B, n_primitives, P, dim - 1)
        r = radius.view(B, self.n_primitives, P, D)

        periodic_net_r = self.get_periodic_net_r(thetas.unsqueeze(1), points,
                                                 r[..., -1])

        final_r = r.clone()
        final_r[..., -1] = r[..., -1] + periodic_net_r.squeeze(-1)

        # B, n_primitives, P, dim
        cartesian_coord = super_shape_functions.polar2cartesian(
            final_r, thetas_reshaped)

        assert [*cartesian_coord.shape] == [B, self.n_primitives, P, self.dim]
        assert not torch.isnan(cartesian_coord).any()

        if self.learn_pose:
            posed_cartesian_coord = self.project_primitive_to_world(
                cartesian_coord, primitive_params)
        else:
            posed_cartesian_coord = cartesian_coord
        # B, n_primitives, P, dim
        return posed_cartesian_coord

    def get_periodic_net_r(self, thetas, points, radius):
        # B, 1 or N, P, dim - 1
        assert len(thetas.shape) == 4, thetas.shape
        assert thetas.shape[-1] == self.dim - 1
        assert points.shape[0] == thetas.shape[0]

        B, _, D = points.shape
        assert points.shape[1] == self.num_points

        _, N, P, Dn1 = thetas.shape
        # B, P, N
        thetas_transposed = thetas.transpose(1, 2).contiguous()

        assert [*radius.shape] == [B, self.n_primitives, P], radius.shape
        radius_transposed = radius.transpose(1, 2).contiguous()

        radius_transposed_repeated = radius_transposed.view(
            B, P, self.n_primitives, 1)
        # B, P, N, dim - 1
        sin = thetas_transposed.sin()
        cos = thetas_transposed.cos()

        # B, P, N, (dim - 1) * 2 = (2 or 4)
        sincos = (torch.cat([sin, cos], axis=-1) * 100).repeat(
            1, 1, self.n_primitives // N, 1)
        assert [*sincos.shape
                ] == [B, P, self.n_primitives,
                      (self.dim - 1) * 2], sincos.shape

        encoded = self.encoder(points).view(B, 1, 1, self.encoder_dim).repeat(
            1, P, self.n_primitives, 1)
        feature_list = [encoded, sincos, radius_transposed_repeated]
        encoded_sincos = torch.cat(feature_list, axis=-1)
        radius = self.decoder(encoded_sincos).view(B, P, self.n_primitives,
                                                   self.num_labels).transpose(
                                                       1, 2).contiguous()

        radius = radius * self.last_scale

        return radius

    def get_indicator(self,
                      x,
                      y,
                      z,
                      r1,
                      r2,
                      theta,
                      phi,
                      *args,
                      points=None,
                      **kwargs):
        assert not points is None
        coord_list = [x, y]
        is3d = len(z.shape) == len(x.shape)
        if is3d:
            # 3D case
            coord_list.append(z)
            angles = torch.stack([theta, phi], axis=-1)
            radius = r2
        else:
            angles = theta.unsqueeze(-1)
            radius = r1
        coord = torch.stack(coord_list, axis=-1)

        rp = self.get_periodic_net_r(angles, points, radius)
        if is3d:
            r2 = r2 + rp.squeeze(-1)
            r2 = nn.functional.relu(r2)
        else:
            r1 = r1 + rp.squeeze(-1)
            r1 = nn.functional.relu(r1)
        numerator = (coord**2).sum(-1)
        denominator = ((r1**2) * (r2**2) * (phi.cos()**2) + (r2**2) *
                       (phi.sin()**2)) + EPS
        indicator = 1. - (numerator / denominator + EPS).sqrt()

        return indicator
