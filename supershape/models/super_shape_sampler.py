import torch
from torch import nn
from layers import super_shape_functions
from models import model_utils


class SuperShapeSampler(nn.Module):
    def __init__(self,
                 max_m,
                 n_primitives,
                 debug=False,
                 rational=True,
                 learn_pose=True,
                 linear_scaling=True):
        super().__init__()
        self.max_m = max_m + 1
        self.n_primitives = n_primitives
        self.rational = rational
        self.learn_pose = learn_pose
        self.linear_scaling = linear_scaling

        # Angle params
        # 1, 1, max_m, 1
        index = torch.arange(start=0, end=max_m + 1, step=1).view(1, 1, -1,
                                                                  1).float()
        # 1, 1, max_m, 1
        self.angles = index / 4.
        assert [*self.angles.shape] == [1, 1, self.max_m,
                                        1], (self.angles.shape, self.max_m)

    def to(self, device):
        super().to(device)
        self.angles = self.angles.to(device)

        return self

    def reshape_params(self, primitive_params):
        """
    Arguments:
      primitive_params (dict): containing supershape's parameters 
    Return:
      reshaped_primitive_params: params with additional dims for easier use in downstream process
    """
        n1, n2, n3, a, b, m_vector, rotation, transition, linear_scale, prob = primitive_params.values(
        )

        assert len(m_vector.shape) == 3
        # 1, n_primitives, max_m, 1
        reshaped_m_vector = m_vector.view(-1, self.n_primitives, self.max_m, 1)

        # Shape params
        assert len(n1.shape) == 2
        # B=1, n_primitives, 1, 1
        reshaped_n1 = n1.view(-1, self.n_primitives, 1, 1)
        reshaped_n2 = n2.view(-1, self.n_primitives, 1, 1)
        reshaped_n3 = n3.view(-1, self.n_primitives, 1, 1)

        # Pose params
        assert len(transition.shape) == 3
        # B=1, n_primitives, 2
        reshaped_transition = transition.view(-1, self.n_primitives, 2)
        assert len(rotation.shape) == 2
        # B=1, n_primitives, 1
        reshaped_rotation = rotation.view(-1, self.n_primitives, 1)
        assert len(linear_scale.shape) == 3
        # B=1, n_primitives, 2
        reshaped_linear_scale = linear_scale.view(-1, self.n_primitives, 2)

        # B=1, n_primitives, 1, 1
        assert len(a.shape) == 2
        reshaped_a = a.view(-1, self.n_primitives, 1, 1)
        reshaped_b = b.view(-1, self.n_primitives, 1, 1)

        return {
            'n1': reshaped_n1,
            'n2': reshaped_n2,
            'n3': reshaped_n3,
            'a': reshaped_a,
            'b': reshaped_b,
            'm_vector': reshaped_m_vector,
            'rotation': reshaped_rotation,
            'transition': reshaped_transition,
            'linear_scale': reshaped_linear_scale,
            'prob': prob
        }

    def transform_circumference_angle_to_super_shape_radius(
        self, thetas, primitive_params, *args, **kwargs):
        """
    Arguments:
      thetas (B, P): Angles between 0 to 2pi
      primitive_params (list or set): containing supershape's parameters 
    Return:
      radius (float): radius value of super shape corresponding to theta
    """
        reshaped_params = self.reshape_params(primitive_params)
        n1 = reshaped_params['n1']
        n2 = reshaped_params['n2']
        n3 = reshaped_params['n3']
        a = reshaped_params['a']
        b = reshaped_params['b']
        m_vector = reshaped_params['m_vector']

        assert len(thetas.shape) == 2
        B = thetas.shape[0]
        P = thetas.shape[-1]

        # B, 1, 1, P
        thetas_dim_added = thetas.view(B, 1, 1, P)
        assert [*thetas_dim_added.shape] == [B, 1, 1, P]

        # B, 1, max_m, P = (1, 1, max_m, 1) x (B, 1, 1, P)
        thetas_max_m = self.angles * thetas_dim_added

        # B, 1, max_m, P
        sin_selected = thetas_max_m.sin()
        cos_selected = thetas_max_m.cos()
        assert [*sin_selected.shape] == [B, 1, self.max_m, P]

        # 1, n_primitives, max_m, 1
        assert [*m_vector.shape] == [1, self.n_primitives, self.max_m, 1]
        assert not torch.isnan(m_vector).any()

        r = self.get_r(thetas_dim_added, sin_selected, cos_selected, n1, n2,
                       n3, a, b, *args, **kwargs) * m_vector
        assert [*r.shape] == [B, self.n_primitives, self.max_m, P]
        assert not torch.isnan(r).any()

        # r = (B, n_primitives, P)
        r = r.sum(2)
        return r

    def get_r(self, thetas_dim_added, sin_selected, cos_selected, n1, n2, n3,
              a, b, *args, **kwargs):
        #r = (B, n_primitives, max_m, P), thetas = (B, 1, 1, P)
        if self.rational:
            r = super_shape_functions.rational_supershape(
                thetas_dim_added, sin_selected, cos_selected, n1, n2, n3, a, b)
        else:
            r = super_shape_functions.supershape(thetas_dim_added,
                                                 sin_selected, cos_selected,
                                                 n1, n2, n3, a, b)
        return r

    def transform_circumference_angle_to_super_shape_world_cartesian_coord(
        self, thetas, radius, primitive_params):
        reshaped_params = self.reshape_params(primitive_params)
        rotation = reshaped_params['rotation']
        transition = reshaped_params['transition']
        linear_scale = reshaped_params['linear_scale']

        assert len(thetas.shape) == 2
        B = thetas.shape[0]
        P = thetas.shape[-1]

        #radius = self.transform_circumference_angle_to_super_shape_radius(thetas, primitive_params)
        assert len(radius.shape) == 3
        # r = (B, n_primitives, P, 1)
        r = radius.view(B, self.n_primitives, P, 1)

        assert [*r.shape] == [B, self.n_primitives, P, 1]

        theta_sincos = thetas.view(B, 1, P)
        assert [*theta_sincos.shape] == [B, 1, P]
        # B, n_primitives, P, 2
        xy = super_shape_functions.rtheta2xy(r, theta_sincos)
        assert [*xy.shape] == [B, self.n_primitives, P, 2]
        assert not torch.isnan(xy).any()

        if self.learn_pose:
            if self.linear_scaling:
                # Scale self.scale=(B=1, n_primitives, 2)
                scaled_xy = xy * linear_scale.view(1, self.n_primitives, 1, 2)
            else:
                scaled_xy = xy
            # B=1, n_primitives, 1
            #rotation = nn.functional.tanh(self.rotation) * math.pi
            # B, n_primitives, P, 2, 2
            rotation_matrix = model_utils.get_rotation_matrix(
                rotation, inv=False).view(1, self.n_primitives, 1, 2,
                                          2).repeat(B, 1, P, 1, 1)
            assert not torch.isnan(rotation_matrix).any()
            # B, n_primitives, P, 2, 1
            xy_before_rotation = scaled_xy.view(B, self.n_primitives, P, 2, 1)
            rotated_xy = torch.bmm(rotation_matrix.view(-1, 2, 2),
                                   xy_before_rotation.view(-1, 2, 1)).view(
                                       B, self.n_primitives, P, 2)
            assert not torch.isnan(rotated_xy).any()
            posed_xy = rotated_xy + transition.view(1, self.n_primitives, 1,
                                                    2)  #.repeat(1, 1, P, 1)
            assert not torch.isnan(posed_xy).any()
        else:
            posed_xy = xy

        # B, n_primitives, P, 2
        return posed_xy

    def transform_world_cartesian_coord_to_tsd(self, xs, ys, primitive_params,
                                               *args, **kwargs):
        reshaped_params = self.reshape_params(primitive_params)
        n1 = reshaped_params['n1']
        n2 = reshaped_params['n2']
        n3 = reshaped_params['n3']
        a = reshaped_params['a']
        b = reshaped_params['b']
        m_vector = reshaped_params['m_vector']
        rotation = reshaped_params['rotation']
        transition = reshaped_params['transition']
        linear_scale = reshaped_params['linear_scale']

        assert len(xs.shape) == 2
        B2 = xs.shape[0]
        P2 = xs.shape[-1]

        # B2, 1, 1, P2
        xs_dim_added = xs.view(B2, 1, 1, P2)
        ys_dim_added = ys.view(B2, 1, 1, P2)
        assert [*xs_dim_added.shape] == [B2, 1, 1, P2]

        if self.learn_pose:
            # B2, P2, 2
            xsys = torch.stack([xs, ys], axis=-1)

            # B2, n_primitives, P2, 2
            xsys_translated = xsys.view(B2, 1, P2, 2) - transition.view(
                1, self.n_primitives, 1, 2)

            # B2, n_primitives, P2, 2, 2
            rotation_matrix2 = model_utils.get_rotation_matrix(
                rotation, inv=True).view(1, self.n_primitives, 1, 2,
                                         2).repeat(B2, 1, P2, 1, 1)
            assert [*rotation_matrix2.shape
                    ] == [B2, self.n_primitives, P2, 2, 2]
            # B2, n_primitives, P2, 2, 1
            xsys_translated_before_rotation = xsys_translated.view(
                B2, self.n_primitives, P2, 2, 1)
            # B2, n_primitives, P2, 2
            rotated_xsys = torch.bmm(
                rotation_matrix2.view(-1, 2, 2),
                xsys_translated_before_rotation.view(-1, 2, 1)).view(
                    B2, self.n_primitives, P2, 2)

            if self.linear_scaling:
                rotated_xsys /= linear_scale.view(1, self.n_primitives, 1, 2)

            # B2, n_primitives, 1, P2
            xs_dim_added = rotated_xsys[:, :, :,
                                        0].view(B2, self.n_primitives, 1, P2)
            ys_dim_added = rotated_xsys[:, :, :,
                                        1].view(B2, self.n_primitives, 1, P2)
        else:
            xs_dim_added = xs_dim_added.repeat(1, self.n_primitives, 1, 1)
            ys_dim_added = ys_dim_added.repeat(1, self.n_primitives, 1, 1)

        sgn = self.get_sgn(xs_dim_added, ys_dim_added, self.angles, n1, n2, n3,
                           a, b, m_vector, *args, **kwargs) * m_vector
        assert [*sgn.shape] == [B2, self.n_primitives, self.max_m, P2]
        # B, n_primitives, P
        output_sgn = sgn.sum(2)
        return output_sgn

    def get_sgn(self, xs_dim_added, ys_dim_added, angles, n1, n2, n3, a, b,
                m_vector, *args, **kwargs):
        if self.rational:
            # B, n_primitives, max_m, P
            sgn = super_shape_functions.implicit_rational_supershape(
                xs_dim_added, ys_dim_added, self.angles, n1, n2, n3, a, b)
        else:
            sgn = super_shape_functions.implicit_supershape(
                xs_dim_added, ys_dim_added, self.angles, n1, n2, n3, a, b)
        return sgn

    def extract_super_shapes_surface_point(self, super_shape_point,
                                           primitive_params, *args, **kwargs):
        """Extract surface point for visualziation purpose"""
        assert len(super_shape_point.shape) == 4
        output_sgn_BxNxNP = self.extract_surface_point_std(
            super_shape_point, primitive_params, *args, **kwargs)
        B, N, P, D = super_shape_point.shape
        B2, N2, NP = output_sgn_BxNxNP.shape
        assert B == 1 and B2 == 1, 'only works with batch size 1'
        surface_mask = self.extract_super_shapes_surface_mask(
            output_sgn_BxNxNP, *args, **kwargs)
        return super_shape_point[surface_mask].view(1, -1, 2)

    def extract_super_shapes_surface_mask(self, output_sgn_BxNxNP, *args,
                                          **kwargs):
        B, N, NP = output_sgn_BxNxNP.shape
        P = NP // N
        output_sgn = output_sgn_BxNxNP.view(B, self.n_primitives,
                                            self.n_primitives, P)
        sgn_p_BxPsN = nn.functional.relu(output_sgn).sum(1).view(
            B, self.n_primitives, P)
        surface_mask = (sgn_p_BxPsN <= 1e-1)
        return surface_mask

    def extract_surface_point_std(self, super_shape_point, primitive_params,
                                  *args, **kwargs):
        B, N, P, D = super_shape_point.shape
        print(B, N, P, D)
        super_shape_point_x = super_shape_point[:, :, :, 0].view(B, N * P)
        super_shape_point_y = super_shape_point[:, :, :, 1].view(B, N * P)

        # B, N, N * P
        output_sgn_BxNxNP = self.transform_world_cartesian_coord_to_tsd(
            super_shape_point_x, super_shape_point_y, primitive_params, *args,
            **kwargs)
        assert [*output_sgn_BxNxNP.shape] == [B, N, N * P]
        return output_sgn_BxNxNP

    def forward(self, primitive_params, thetas=None, xs=None, ys=None):
        if thetas is not None:
            # B, N, P
            radius = self.transform_circumference_angle_to_super_shape_radius(
                thetas, primitive_params)
            # B, N, P, 2
            super_shape_point = self.transform_circumference_angle_to_super_shape_world_cartesian_coord(
                thetas, radius, primitive_params)

            output_sgn_BxNxNP = self.extract_surface_point_std(
                super_shape_point, primitive_params)
            # B, P', 2
            surface_mask = self.extract_super_shapes_surface_mask(
                output_sgn_BxNxNP)
        else:
            super_shape_point = None
            surface_mask = None

        if xs is not None and ys is not None:
            tsd = self.transform_world_cartesian_coord_to_tsd(
                xs, ys, primitive_params)
        else:
            tsd = None

        # (B, N, P, 2), (B, N, P), (B, N, P2)
        return super_shape_point, surface_mask, tsd
