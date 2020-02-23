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
                 linear_scaling=True,
                 dim=2):
        """Intitialize SuperShapeSampler.

        Args:
            max_m: Max number of vertices in a primitive. If quadrics is True,
                then max_m must be larger than 4.
            n_primitives: Number of primitives to use.
            debug: debug mode.
            rational: Use rational (stable gradient) version of supershapes
            learn_pose: Apply pose change flag
            linear_scaling: linearly scale the shape flag
            dim: if dim == 2, then 2D mode. If 3, 3D mode.
        """
        super().__init__()
        self.max_m = max_m + 1
        self.n_primitives = n_primitives
        self.rational = rational
        self.learn_pose = learn_pose
        self.linear_scaling = linear_scaling

        self.dim = dim
        if not self.dim in [2, 3]:
            raise NotImplementedError('dim must be either 2 or 3.')

        # Angle params
        # 1, 1, max_m, 1, 1
        index = torch.arange(start=0, end=max_m + 1,
                             step=1).view(1, 1, -1, 1, 1).float()
        # 1, 1, max_m, 1, 1
        self.angles = index / 4.
        assert [*self.angles.shape] == [1, 1, self.max_m, 1,
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

        assert len(m_vector.shape) == 4
        # 1, n_primitives, max_m, 1, dim-1
        reshaped_m_vector = m_vector.view(-1, self.n_primitives, self.max_m, 1,
                                          self.dim - 1)

        # Shape params
        assert len(n1.shape) == 3
        # B=1, n_primitives, 1, 1, dim - 1
        reshaped_n1 = n1.view(-1, self.n_primitives, 1, 1, self.dim - 1)
        reshaped_n2 = n2.view(-1, self.n_primitives, 1, 1, self.dim - 1)
        reshaped_n3 = n3.view(-1, self.n_primitives, 1, 1, self.dim - 1)

        # Pose params
        assert len(transition.shape) == 3
        # B=1, n_primitives, dim
        reshaped_transition = transition.view(-1, self.n_primitives, self.dim)
        assert len(rotation.shape) == 3
        # B=1, n_primitives, 1
        if self.dim == 2:
            reshaped_rotation = rotation.view(-1, self.n_primitives, 1)
        else:
            reshaped_rotation = rotation.view(-1, self.n_primitives, 4)
        assert len(linear_scale.shape) == 3
        # B=1, n_primitives, dim
        reshaped_linear_scale = linear_scale.view(-1, self.n_primitives,
                                                  self.dim)

        # B=1, n_primitives, 1, 1, self.dim - 1
        assert len(a.shape) == 3
        reshaped_a = a.view(-1, self.n_primitives, 1, 1, self.dim - 1)
        reshaped_b = b.view(-1, self.n_primitives, 1, 1, self.dim - 1)

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

        assert len(thetas.shape) == 3
        B = thetas.shape[0]
        P = thetas.shape[1]
        D = thetas.shape[2]
        assert D == self.dim - 1

        # B, 1, 1, P
        thetas_dim_added = thetas.view(B, 1, 1, P, D)
        assert [*thetas_dim_added.shape] == [B, 1, 1, P, D]

        # B, 1, max_m, P, D = (1, 1, max_m, 1, 1) x (B, 1, 1, P, D)
        thetas_max_m = self.angles * thetas_dim_added

        # B, 1, max_m, P, D
        sin_selected = thetas_max_m.sin()
        cos_selected = thetas_max_m.cos()
        assert [*sin_selected.shape] == [B, 1, self.max_m, P, D]

        # 1, n_primitives, max_m, 1, 1
        assert [*m_vector.shape] == [1, self.n_primitives, self.max_m, 1, D]
        assert not torch.isnan(m_vector).any()

        r = self.get_r(thetas_dim_added, sin_selected, cos_selected, n1, n2,
                       n3, a, b, *args, **kwargs) * m_vector
        assert [*r.shape] == [B, self.n_primitives, self.max_m, P, D]
        assert not torch.isnan(r).any()

        # r = (B, n_primitives, P, D)
        r = r.sum(2)
        return r

    def get_r(self, thetas_dim_added, sin_selected, cos_selected, n1, n2, n3,
              a, b, *args, **kwargs):
        #r = (B, n_primitives, max_m, P, dim-1), thetas = (B, 1, 1, P, dim-1)
        if self.rational:
            r = super_shape_functions.rational_supershape(
                thetas_dim_added, sin_selected, cos_selected, n1, n2, n3, a, b)
        else:
            r = super_shape_functions.supershape(thetas_dim_added,
                                                 sin_selected, cos_selected,
                                                 n1, n2, n3, a, b)
        return r

    def transform_circumference_angle_to_super_shape_world_cartesian_coord(
        self, angles, radius, primitive_params):
        reshaped_params = self.reshape_params(primitive_params)
        rotation = reshaped_params['rotation']
        transition = reshaped_params['transition']
        linear_scale = reshaped_params['linear_scale']

        assert len(angles.shape) == 3
        B, P, D = angles.shape

        #radius = self.transform_circumference_angle_to_super_shape_radius(thetas, primitive_params)
        assert len(radius.shape) == 4
        # r = (B, n_primitives, P, dim - 1)
        r = radius.view(B, self.n_primitives, P, D)

        angles_reshaped = angles.view(B, 1, P, D)
        # B, n_primitives, P, dim
        cartesian_coord = super_shape_functions.polar2cartesian(
            r, angles_reshaped)
        assert [*cartesian_coord.shape] == [B, self.n_primitives, P, self.dim]
        assert not torch.isnan(cartesian_coord).any()

        if self.learn_pose:
            if self.linear_scaling:
                # Scale self.scale=(B=1, n_primitives, dim)
                scaled_cartesian_coord = cartesian_coord * linear_scale.view(
                    1, self.n_primitives, 1, self.dim)
            else:
                scaled_cartesian_coord = cartesian_coord

            rotated_cartesian_coord = model_utils.apply_rotation(
                scaled_cartesian_coord, rotation)
            assert not torch.isnan(rotated_cartesian_coord).any()
            posed_cartesian_coord = rotated_cartesian_coord + transition.view(
                1, self.n_primitives, 1, self.dim)
            assert not torch.isnan(posed_cartesian_coord).any()
        else:
            posed_cartesian_coord = cartesian_coord

        # B, n_primitives, P, dim
        return posed_cartesian_coord

    def transform_world_cartesian_coord_to_tsd(self, coord, primitive_params,
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

        assert len(coord.shape) == 3, coord.shape
        B, P, D = coord.shape
        assert D == self.dim

        if self.learn_pose:
            # B, n_primitives, P, 2
            coord_translated = coord.view(B, 1, P, D) - transition.view(
                1, self.n_primitives, 1, D)
            rotated_coord = model_utils.apply_rotation(coord_translated,
                                                       rotation,
                                                       inv=True)

            if self.linear_scaling:
                rotated_coord /= linear_scale.view(1, self.n_primitives, 1, D)

            # B, n_primitives, 1, P, D
            coord_dim_added = rotated_coord.view(B, self.n_primitives, 1, P, D)
        else:
            coord_dim_added = coord.view(B, 1, 1, P,
                                         D).repeat(1, self.n_primitives, 1, 1,
                                                   1)

        print(m_vector.shape)
        sgn = self.get_sgn(coord_dim_added, self.angles, n1, n2, n3, a, b,
                           m_vector, *args, **kwargs) * m_vector
        assert [*sgn.shape] == [B, self.n_primitives, self.max_m, P]
        # B, n_primitives, P
        output_sgn = sgn.sum(2)
        return output_sgn

    def get_sgn(self, coord_dim_added, angles, n1, n2, n3, a, b, m_vector,
                *args, **kwargs):
        if self.rational:
            # B, n_primitives, max_m, P
            sgn = super_shape_functions.implicit_rational_supershape(
                coord_dim_added, self.angles, n1, n2, n3, a, b)
        else:
            raise NotImplementedError('implicit supershape not supported yet.')
            sgn = super_shape_functions.implicit_supershape(
                coord_dim_added, self.angles, n1, n2, n3, a, b)
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
        return super_shape_point[surface_mask].view(1, -1, D)

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

    def forward(self, primitive_params, thetas=None, coord=None):
        if thetas is not None:
            # B, N, P
            radius = self.transform_circumference_angle_to_super_shape_radius(
                thetas, primitive_params)
            # B, N, P, dim
            super_shape_point = self.transform_circumference_angle_to_super_shape_world_cartesian_coord(
                thetas, radius, primitive_params)

            output_sgn_BxNxNP = self.extract_surface_point_std(
                super_shape_point, primitive_params)
            # B, P', dim
            surface_mask = self.extract_super_shapes_surface_mask(
                output_sgn_BxNxNP)
        else:
            super_shape_point = None
            surface_mask = None

        if coord is not None:
            tsd = self.transform_world_cartesian_coord_to_tsd(
                coord, primitive_params)
        else:
            tsd = None

        # (B, N, P, dim), (B, N, P), (B, N, P2)
        return super_shape_point, surface_mask, tsd
