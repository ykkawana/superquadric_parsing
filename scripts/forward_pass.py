#!/usr/bin/env python
"""Script used to perform a forward pass using a previously trained model and
visualize the corresponding primitives
"""
from itertools import product
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
from pyrender.constants import RenderFlags
import pyrender
import trimesh
import skimage.io
import pickle
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from arguments import add_voxelizer_parameters, add_nn_parameters, \
     add_dataset_parameters, add_gaussian_noise_layer_parameters, \
     voxelizer_shape, add_loss_options_parameters, add_loss_parameters
from utils import get_colors, store_primitive_parameters
from visualization_utils import points_on_sq_surface, points_on_cuboid, \
    save_prediction_as_ply, get_primitive_trimesh, get_prediction_as_trimesh

from learnable_primitives.common.dataset import get_dataset_type,\
    compose_transformations
from learnable_primitives.common.model_factory import DatasetBuilder
from learnable_primitives.equal_distance_sampler_sq import\
    EqualDistanceSamplerSQ
from learnable_primitives.models import NetworkParameters
from learnable_primitives.loss_functions import euclidean_dual_loss
from learnable_primitives.primitives import\
    euler_angles_to_rotation_matrices, quaternions_to_rotation_matrices
from learnable_primitives.voxelizers import VoxelizerFactory


#from mayavi import mlab

def load_model(model, args, device):
    layer_map_ori = {
        "_primitive_layer.layer0._translation_layer.weight": "_primitive_layer.layer1._translation_layer.weight",
        "_primitive_layer.layer0._translation_layer.bias": "_primitive_layer.layer1._translation_layer.bias",
        "_primitive_layer.layer1._rotation_layer.weight": "_primitive_layer.layer2._rotation_layer.weight",
        "_primitive_layer.layer1._rotation_layer.bias": "_primitive_layer.layer2._rotation_layer.bias",
        "_primitive_layer.layer2._size_layer.weight": "_primitive_layer.layer0._size_layer.weight",
        "_primitive_layer.layer2._size_layer.bias": "_primitive_layer.layer0._size_layer.bias",
        "_primitive_layer.layer3._probability_layer.weight": "_primitive_layer.layer5._probability_layer.weight",
        "_primitive_layer.layer3._probability_layer.bias": "_primitive_layer.layer5._probability_layer.bias",
        "_primitive_layer.layer5._tapering_layer.weight": "_primitive_layer.layer3._tapering_layer.weight",
        "_primitive_layer.layer5._tapering_layer.bias": "_primitive_layer.layer3._tapering_layer.bias",
        "_primitive_layer.layer4._shape_layer.weight": "_primitive_layer.layer4._shape_layer.weight",
        "_primitive_layer.layer4._shape_layer.bias": "_primitive_layer.layer4._shape_layer.bias"
    }
    layer_map = {layer_map_ori[key]: key for key in layer_map_ori}
    #layer_map = layer_map_ori
    loaded = torch.load(args.weight_file, map_location=device)
    loaded_copied = {} 
    for key in loaded:
        print(key)
    for key, values in model.named_parameters():
        print(key)
    for key in loaded:
        values = loaded[key]
        #if key.startswith('_features_') or key.startswith('_primitive_layer.layer4._shape_layer.'):
        #    loaded_copied[key] = loaded[key]
        print(key)
        if key.startswith('_primitive_layer'):
            loaded_copied[layer_map[key]] = loaded[key]
            #loaded_copied[key] = loaded[layer_map[key]]
        #elif key.endswith('running_mean') or key.endswith('running_var') or key.endswith('tracked'):
        #    loaded_copied[key] = values
        else:
            loaded_copied[key] = loaded[key]
    for key in loaded_copied:
        print(key)
    model.load_state_dict(loaded_copied)
    return model

def get_shape_configuration(use_cuboids):
    if use_cuboids:
        return points_on_cuboid
    else:
        return points_on_sq_surface

def resize(ratio):
    return np.array([
        [ratio, 0,   0,   0],
        [0,  ratio, 0.0, 0],
        [0,  0,   ratio,   0],
        [0.0,  0.0, 0.0, 1],
    ])


def translate(x=0, y=0, z=0):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ])


def rotate(x=0, y=0, z=0):
    xrd = x/180*np.pi
    yrd = y/180*np.pi
    zrd = z/180*np.pi

    xr = np.array([
        [1, 0, 0, 0],
        [0, np.cos(xrd), -np.sin(xrd), 0],
        [0, np.sin(xrd),   np.cos(xrd), 0],
        [0, 0, 0, 1],
    ])

    yr = np.array([
        [np.cos(yrd), 0, np.sin(yrd), 0],
        [0, 1, 0, 0],
        [-np.sin(yrd),  0, np.cos(yrd), 0],
        [0, 0, 0, 1],
    ])

    zr = np.array([
        [np.cos(zrd), -np.sin(zrd), 0, 0],
        [np.sin(zrd),  np.cos(zrd), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return np.dot(zr, np.dot(yr, xr))


def transform(trans):
    res = trans[0]
    for idx in range(len(trans)-1):
        res = np.dot(trans[idx+1], res)
    return res



def main(argv):
    parser = argparse.ArgumentParser(
        description="Do the forward pass and estimate a set of primitives"
    )
    parser.add_argument(
        "dataset_directory",
        help="Path to the directory containing the dataset"
    )
    parser.add_argument(
        "output_directory",
        help="Save the output files in that directory"
    )
    parser.add_argument(
        "--tsdf_directory",
        default="",
        help="Path to the directory containing the precomputed tsdf files"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="The path to the previously trainined model to be used"
    )

    parser.add_argument(
        "--n_primitives",
        type=int,
        default=32,
        help="Number of primitives"
    )
    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold"
    )
    parser.add_argument(
        "--use_deformations",
        action="store_true",
        help="Use Superquadrics with deformations as the shape configuration"
    )
    parser.add_argument(
        "--save_prediction_as_mesh",
        action="store_true",
        help="When true store prediction as a mesh"
    )
    parser.add_argument(
        "--run_on_gpu",
        action="store_true",
        help="Use GPU"
    )
    parser.add_argument(
        "--with_animation",
        action="store_true",
        help="Add animation"
    )

    add_dataset_parameters(parser)
    add_nn_parameters(parser)
    add_voxelizer_parameters(parser)
    add_gaussian_noise_layer_parameters(parser)
    add_loss_parameters(parser)
    add_loss_options_parameters(parser)
    args = parser.parse_args(argv)

    # A sampler instance
    e = EqualDistanceSamplerSQ(200)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    if args.run_on_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on ", device)

    # Create a factory that returns the appropriate voxelizer based on the
    # input argument
    voxelizer_factory = VoxelizerFactory(
        args.voxelizer_factory,
        np.array(voxelizer_shape(args)),
        args.save_voxels_to
    )

    # Create a dataset instance to generate the samples for training
    dataset = get_dataset_type("euclidean_dual_loss")(
        (DatasetBuilder()
            .with_dataset(args.dataset_type)
            .filter_tags(args.model_tags)
            .build(args.dataset_directory)),
        voxelizer_factory,
        args.n_points_from_mesh,
        transform=compose_transformations(voxelizer_factory)
    )

    # TODO: Change batch_size in dataloader
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

    network_params = NetworkParameters.from_options(args)
    # Build the model to be used for testing
    model = network_params.network(network_params)
    # Move model to device to be used
    model.to(device)

    #model = load_model(model, args, device)
    #torch.save(model.state_dict(), 'chair_T26AK2FES_model_699_py3')

    if args.weight_file is not None:
        # Load the model parameters of the previously trained model
        model.load_state_dict(
            torch.load(args.weight_file, map_location=device)
        )
    model.eval()

    colors = get_colors(args.n_primitives)
    for sample in dataloader:
        X, y_target = sample
        X, y_target = X.to(device), y_target.to(device)

        # Do the forward pass and estimate the primitive parameters
        y_hat = model(X)

        M = args.n_primitives  # number of primitives
        probs = y_hat[0].to("cpu").detach().numpy()
        # Transform the Euler angles to rotation matrices
        if y_hat[2].shape[1] == 3:
            R = euler_angles_to_rotation_matrices(
                y_hat[2].view(-1, 3)
            ).to("cpu").detach()
        else:
            R = quaternions_to_rotation_matrices(
                    y_hat[2].view(-1, 4)
                ).to("cpu").detach()
            # get also the raw quaternions
            quats = y_hat[2].view(-1, 4).to("cpu").detach().numpy()
        translations = y_hat[1].to("cpu").view(args.n_primitives, 3)
        translations = translations.detach().numpy()

        shapes = y_hat[3].to("cpu").view(args.n_primitives, 3).detach().numpy()
        epsilons = y_hat[4].to("cpu").view(
            args.n_primitives, 2
        ).detach().numpy()
        taperings = y_hat[5].to("cpu").view(
            args.n_primitives, 2
        ).detach().numpy()

        pts = y_target[:, :, :3].to("cpu")
        pts_labels = y_target[:, :, -1].to("cpu").squeeze().numpy()
        pts = pts.squeeze().detach().numpy().T

        on_prims = 0
        """
        fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        mlab.view(azimuth=0.0, elevation=0.0, distance=2)
        """
        # Uncomment to visualize the points sampled from the target mesh
        # t = np.array([1.2, 0, 0]).reshape(3, -1)
        # pts_n = pts + t
        #     mlab.points3d(
        #        # pts_n[0], pts_n[1], pts_n[2],
        #        pts[0], pts[1], pts[2],
        #        scale_factor=0.03, color=(0.8, 0.8, 0.8)
        #     )

        # Keep track of the files containing the parameters of each primitive

        scene = pyrender.Scene(ambient_light=[1., 1., 1.])
        pose = np.array([
            [1, 0,   0,   0],
            [0,  1, 0.0, 0],
            [0,  0,   1,   -1],
            [0.0,  0.0, 0.0, 1.0],
        ])

        f = 1469.333
        w = 1280
        h = 768
        cx = w//2
        cy = h//2
        #camera = pyrender.PerspectiveCamera(yfov=60/180*np.pi, aspectRatio=1.0)
        camera = pyrender.IntrinsicsCamera(f, f, cx, cy)

        camera_pose = np.array([
            [1, 0,   0,   -1],
            [0,  1, 0.0, -1],
            #[0,  0,   1,   -1], #-0.5
            [0,  0,   1,   -0.5],  # -0.5
            [0.0,  0.0, 0.0, 1.0],
        ])
        camera_pose = transform([camera_pose, rotate(x=-45)])

        scene.add(camera, pose=camera_pose)


        primitive_files = []
        for i in range(args.n_primitives):
            x_tr, y_tr, z_tr, prim_pts =\
                get_shape_configuration(args.use_cuboids)(
                    shapes[i, 0],
                    shapes[i, 1],
                    shapes[i, 2],
                    epsilons[i, 0],
                    epsilons[i, 1],
                    R[i].numpy(),
                    translations[i].reshape(-1, 1),
                    taperings[i, 0],
                    taperings[i, 1]
                )

            # Dump the parameters of each primitive as a dictionary
            store_primitive_parameters(
                size=tuple(shapes[i]),
                shape=tuple(epsilons[i]),
                rotation=tuple(quats[i]),
                location=tuple(translations[i]),
                tapering=tuple(taperings[i]),
                probability=(probs[0, i],),
                color=(colors[i % len(colors)]) + (1.0,),
                filepath=os.path.join(
                    args.output_directory,
                    "primitive_%d.p" %(i,)
                )
            )
            if probs[0, i] >= args.prob_threshold:
                on_prims += 1
          
                """
                mlab.mesh(
                    x_tr,
                    y_tr,
                    z_tr,
                    color=tuple(colors[i % len(colors)]),
                    opacity=1.0
                )
                """
                primitive_files.append(
                    os.path.join(args.output_directory, "primitive_%d.p" % (i,))
                )

        if args.with_animation:
            cnt = 0
            for az in range(0, 360, 1):
                cnt += 1
                mlab.view(azimuth=az, elevation=0.0, distance=2)
                mlab.savefig(
                    os.path.join(
                        args.output_directory,
                        "img_%04d.png" % (cnt,)
                    )
                )
        for i in range(args.n_primitives):
            print(i, probs[0, i])

        print("Using %d primitives out of %d" % (on_prims, args.n_primitives))

        print('start rendering...')
        scene.add(pyrender.Mesh.from_trimesh(get_prediction_as_trimesh(
                primitive_files
            ), smooth=False))
        r = pyrender.OffscreenRenderer(w, h)
        flags = RenderFlags.SKIP_CULL_FACES  # RGBA | RenderFlags.SHADOWS_DIRECTIONAL
        color, depth = r.render(scene, flags)
        plt.figure(figsize=(16, 12))
        plt.axis('off')
        plt.imshow(color)
        skimage.io.imsave(
            os.path.join(args.output_directory, "rendered.png"),
            color)

        if args.save_prediction_as_mesh:
            print("Saving prediction as mesh....")
            save_prediction_as_ply(
                primitive_files,
                os.path.join(args.output_directory, "primitives.ply")
            )
            print("Saved prediction as ply file in {}".format(
                os.path.join(args.output_directory, "primitives.ply")
            ))


if __name__ == "__main__":
    main(sys.argv[1:])
