import torch
from torch import nn
from layers import primitive_wise_layers
from models import super_shape_sampler
import utils
from layers import layer_utils


class PeriodicShapeSampler(super_shape_sampler.SuperShapeSampler):
    def __init__(self,
                 num_points,
                 max_m,
                 n_primitives=1,
                 last_bias=1.,
                 rational=True,
                 learn_pose=True,
                 linear_scaling=True,
                 dim=2,
                 mode='scratch',
                 factor=1,
                 act='leaky',
                 *args,
                 **kwargs):
        super().__init__(max_m,
                         n_primitives,
                         rational=rational,
                         learn_pose=learn_pose,
                         linear_scaling=linear_scaling,
                         *args,
                         **kwargs)
        assert dim == 2, 'currently only 2d is supported'
        self.factor = factor
        self.num_points = num_points
        self.num_labels = 1
        self.dim = dim
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
        self.n_primitives = n_primitives
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
                                                      c64 * 2 + self.dim,
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

    def forward(self, points, primitive_params, thetas=None, xs=None, ys=None):
        if thetas is not None:
            # B, N, P
            radius = self.transform_circumference_angle_to_super_shape_radius(
                thetas, primitive_params, points=points)
            # B, N, P, 2
            super_shape_point = self.transform_circumference_angle_to_super_shape_world_cartesian_coord(
                thetas, radius, primitive_params)

            output_sgn_BxNxNP = self.extract_surface_point_std(
                super_shape_point, primitive_params, points=points)
            # B, P', 2
            surface_mask = self.extract_super_shapes_surface_mask(
                output_sgn_BxNxNP)
        else:
            super_shape_point = None
            surface_mask = None

        if xs is not None and ys is not None:
            tsd = self.transform_world_cartesian_coord_to_tsd(xs,
                                                              ys,
                                                              primitive_params,
                                                              points=points)
        else:
            tsd = None

        # (B, N, P, 2), (B, N, P), (B, N, P2)
        return super_shape_point, surface_mask, tsd

    def get_r(self, thetas, sin_selected, cos_selected, n1, n2, n3, a, b,
              *args, **kwargs):
        points = kwargs['points']
        # B, P, 3 or 2
        assert len(points.shape) == 3
        assert points.shape[1] == self.num_points
        # B, 1, 1, P2
        assert len(thetas.shape) == 4
        assert points.shape[0] == thetas.shape[0]
        B, P, D = points.shape
        B, N, _, P2 = thetas.shape
        # B, P2, N
        thetas_transposed = thetas.view(B, N, P2).transpose(1, 2).contiguous()

        sin = thetas_transposed.sin()
        cos = thetas_transposed.cos()
        sincos = torch.stack([sin, cos], axis=-1) * 100
        points_BxPxNx2or3 = points.view(B, P, 1, self.dim).repeat(
            1, 1, self.n_primitives, 1)
        # Encoder expects B, P, N, D -> B, 1, N, self.encoder_dim
        encoded = self.encoder(points_BxPxNx2or3)
        # sincos (B, P2, 2)
        concated_BxP2xNx2or3pencoder_dim = torch.cat([
            encoded.view(B, 1, self.n_primitives, self.encoder_dim).repeat(
                1, P2, 1, 1),
            sincos.repeat(1, 1, self.n_primitives // N, 1)
        ],
                                                     axis=-1)
        radius = self.decoder(concated_BxP2xNx2or3pencoder_dim).view(
            B, P2, self.n_primitives).transpose(1, 2).contiguous()
        radius = radius * self.last_scale
        if self.mode == 'scratch':
            radius = radius + self.last_bias
            radius = nn.functional.relu(radius) + 1e-7
            radius = radius.view(B, self.n_primitives, 1,
                                 P2).repeat(1, 1, self.max_m, 1)
        elif self.mode == 'delta':
            radius = radius.unsqueeze(2) + super().get_r(
                thetas, sin_selected, cos_selected, n1, n2, n3, a, b, *args, **
                kwargs)

            radius = nn.functional.relu(radius) + 1e-7
        else:
            raise NotImplementedError('Mode mush be one of scratch or delta')
        return radius

    def get_sgn(self, x, y, angles, n1, n2, n3, a, b, *args, **kwargs):
        points = kwargs['points']
        theta = utils.safe_atan(y, x)
        B, _, _, P = theta.shape
        # B, 1, max_m, P = (1, 1, max_m, 1) x (B, N, 1, P)
        thetas_max_m = self.angles * theta
        # B, 1, max_m, P
        sin_selected = thetas_max_m.sin()
        cos_selected = thetas_max_m.cos()
        assert [*sin_selected.shape] == [B, self.n_primitives, self.max_m, P]
        r = self.get_r(theta,
                       sin_selected,
                       cos_selected,
                       n1,
                       n2,
                       n3,
                       a,
                       b,
                       points=points)
        assert [*r.shape] == [B, self.n_primitives, self.max_m, P]

        indicator = layer_utils.get_indicator(x, y, r)
        return indicator
