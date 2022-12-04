#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script for training multiplane image (MPI) network.
"""

from __future__ import division
import tensorflow as tf
import numpy as np
import argparse
import os

from mpi_extrapolation.mpi_with_stylization import MPI
from WCT_TF.WCT_TF.utils import get_img

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_dir',
                    type=str,
                    default='mpi_extrapolation/models/',
                    help='Location to save the models.')
parser.add_argument('--inputs',
                    type=str,
                    default='mpi_extrapolation/examples/1.npz',
                    help='Directory of dateset.')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.0002,
                    help='Learning rate')
parser.add_argument('--max_steps',
                    type=int,
                    default=10000000,
                    help='Maximum number of training steps.')
parser.add_argument('--summary_freq',
                    type=int,
                    default=10,
                    help='Logging frequency.')
parser.add_argument('--vgg_path_for_wct',
                    type=str,
                    help='Path to vgg_normalised.t7',
                    default='WCT_TF/WCT_TF/models/vgg_normalised.t7')
parser.add_argument('--style_img_path',
                    type=str,
                    help='path to one style image.',
                    default='images/style_sunrise.png')
parser.add_argument('--device',
                    type=str,
                    help='Device to perform compute on, e.g. /gpu:0',
                    default='/cpu:0')
parser.add_argument('--wct_model_dirs',
                    nargs='+',
                    type=str,
                    help='List of WCT checkpoint directories',
                    default=[
                        'WCT_TF/WCT_TF/models/relu5_1',
                        'WCT_TF/WCT_TF/models/relu4_1',
                        'WCT_TF/WCT_TF/models/relu3_1',
                        'WCT_TF/WCT_TF/models/relu2_1',
                        'WCT_TF/WCT_TF/models/relu1_1'
                    ])
parser.add_argument(
    '--save_latest_freq',
    type=int,
    default=100,
    help='Frequency with which to save the model (overwrites previous model).')
parser.add_argument(
    '--vgg_model_file',
    type=str,
    default='/data/styletrans/imagenet-vgg-verydeep-19.mat',
    help='Location of vgg model file used to compute perceptual (VGG) loss.')
parser.add_argument(
    '--wct_relu_targets',
    nargs='+',
    type=str,
    help='List of reluX_1 layers, corresponding to --checkpoints',
    default=['relu5_1', 'relu4_1', 'relu3_1', 'relu2_1', 'relu1_1'])

args = parser.parse_args()

# Continue training from previous checkpoint.
Continue_train = False
# Number of source images.
Num_source = 2
# Minimum scene depth.
Min_depth = 1
# Maximum scene depth.
Max_depth = 100
# Number of planes for plane sweep volume (PSV).
Num_psv_planes = 32
# Number of MPI planes to predict.
Num_mpi_planes = 32


def convert_from_npz(inputs):
    input_dict = {}
    input_dict["src_images"] = tf.constant(inputs["src_images"])
    input_dict["ref_image"] = tf.constant(inputs["ref_image"])
    input_dict["ref_pose"] = tf.constant(inputs["ref_pose"])
    input_dict["src_poses"] = tf.constant(inputs["src_poses"])
    input_dict["intrinsics"] = tf.constant(inputs["intrinsics"])
    input_dict["tgt_pose"] = tf.constant(inputs["tgt_pose"])
    input_dict["tgt_image"] = tf.constant(inputs["tgt_image"])

    input_dict["ref_image"] = tf.image.convert_image_dtype(
        input_dict["ref_image"], dtype=tf.float32)
    input_dict["src_images"] = tf.image.convert_image_dtype(
        input_dict["src_images"], dtype=tf.float32)
    input_dict["tgt_image"] = tf.image.convert_image_dtype(
        input_dict["tgt_image"], dtype=tf.float32)

    return input_dict


def train():
    tf.compat.v1.logging.set_verbosity(tf.logging.INFO)

    if not tf.gfile.IsDirectory(args.checkpoint_dir):
        tf.gfile.MakeDirs(args.checkpoint_dir)
        print(f'creating directory {args.checkpoint_dir}')

    if not os.path.isfile(args.vgg_model_file):
        print("`vgg_model_file` doesn't exist.")
        return

    inputs = convert_from_npz(np.load(args.inputs))

    model = MPI()
    train_op = model.build_train_graph(inputs=inputs,
                                       min_depth=Min_depth,
                                       max_depth=Max_depth,
                                       style_img_path=args.style_img_path,
                                       num_mpi_planes=Num_mpi_planes,
                                       learning_rate=args.learning_rate,
                                       wct_model_dirs=args.wct_model_dirs,
                                       wct_relu_targets=args.wct_relu_targets,
                                       vgg_path_for_wct=args.vgg_path_for_wct,
                                       vgg_model_file=args.vgg_model_file,
                                       device_wct=args.device,
                                       ss_patch_size=3,
                                       ss_stride=1,
                                       beta1=0.9)

    model.train(train_op=train_op,
                load_dir=args.checkpoint_dir,
                checkpoint_dir=args.checkpoint_dir,
                summary_dir=args.checkpoint_dir,
                continue_train=Continue_train,
                summary_freq=args.summary_freq,
                save_latest_freq=args.save_latest_freq,
                max_steps=args.max_steps)


if __name__ == '__main__':
    train()