import torch


def custom_chamfer_loss(primitive_points,
                        target_points,
                        prob=None,
                        surface_mask=None):
    """
  N = n_primitives
  B = batch
  Ps = points per primitive
  Pt = target points

  primitive_points: B x N x Ps x 2
  target_points: B x Pt x 2
  prob: B x N
  """
    assert len(primitive_points.shape) == 4
    assert len(target_points.shape) == 3

    B, N, Ps, _ = primitive_points.shape
    _, Pt, _ = target_points.shape

    # B x 1 x (Ps * N) x 2
    primitive_points_Bx1xPsNx2 = primitive_points.view(B, -1, 2).unsqueeze(1)

    # B x Pt x 1 x 2
    target_points_BxPtx1x2 = target_points.unsqueeze(2)

    # B x Pt x (Ps * N) x 2
    diff_target2primitives = target_points_BxPtx1x2 - primitive_points_Bx1xPsNx2

    # B x Pt x (Ps * N)
    dist_target2primitives = torch.norm(diff_target2primitives, 2, dim=3)

    # B x Pt
    loss_target2primitives = torch.min(dist_target2primitives, dim=2)[0].mean()

    # B x (Ps * N) x 1 x 2
    primitive_points_BxPsNx1x2 = primitive_points.view(B, -1, 2).unsqueeze(2)

    # B x 1x Pt x 2
    target_points_Bx1xPtx2 = target_points.unsqueeze(1)

    # B x (Ps * N) x Pt x 2
    diff_primitives2target = primitive_points_BxPsNx1x2 - target_points_Bx1xPtx2

    # B x (Ps * N) x Pt
    dist_primitives2target = torch.norm(diff_primitives2target, 2, dim=3)

    # B x (Ps * N)
    loss_primitives2target = torch.min(dist_primitives2target, dim=2)[0]

    if prob is not None:
        assert len(prob.shape) == 2
        # B x N x Ps
        prob_BxNxPs = prob.view(1, -1, 1).repeat(1, 1, Ps)

        # B x (Ps * N)
        prob_BxPsN = prob_BxNxPs.view(1, -1)

        loss_primitives2target *= prob_BxPsN

    if surface_mask is not None:
        # B, N, Ps
        assert len(surface_mask.shape) == 3
        assert [*primitive_points.shape[:-1]] == [*surface_mask.shape
                                                  ], (primitive_points.shape,
                                                      surface_mask.shape)

        # B x (Ps * N)
        surface_mask_BxPsN = surface_mask.view(B, -1)

        loss_primitives2target *= surface_mask_BxPsN

    loss_primitives2target = loss_primitives2target.mean()

    return (loss_primitives2target + loss_target2primitives) / 2.
