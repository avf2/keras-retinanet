#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import json

import argparse
import os
import sys
import pandas as pd
import numpy as np

import keras
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..preprocessing.csv_generator import CSVGenerator
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.eval import _get_detections
from ..utils.keras_version import check_keras_version


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args):
    """ Create generators for image f.
    """
    feeding_generator = CSVGenerator(
        args.data_img_list,
        args.data_class_map,
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side,
        config=args.config
    )
    return feeding_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Script for running inference with pretrained RetinaNet network.')

    parser.add_argument('--data-img-list',    required=True, help='Path to CSV file containing paths to images to be processed.')
    parser.add_argument('--data-class-map',   required=True, help='Path to a CSV file containing class label mapping.')

    parser.add_argument('--model',            required=True, help='Path to RetinaNet model.')
    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')
    parser.add_argument('--save-path',        help='Path for saving outputs.', default='/tmp/retinanet/output')
    parser.add_argument('--predictions-filename',  required=True, help='Filename for saving outputs.')
    parser.add_argument('--class-map-filename',    required=True, help='Filename for class-map used.')
    parser.add_argument('--save-output-imgs',      help='If True, images will be saved with predictions drawn on them.', action='store_true')
    parser.add_argument('--inner-features-file',   help='If not None, features from an inner layer of the model will be stored in this file.', default=None)
    parser.add_argument('--inner-features-layer',  help='If --save-inner-features-file is not None, name of the layer whose features should be stored', default='P7')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    if args.inner_features_file is not None:
        assert args.inner_features_file.endswith('.npz'), "inner_features_file's extension should be .npz"
        get_inner_features = True
    else:
        get_inner_features = False

    # create the generator
    generator = create_generator(args)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone)

    # optionally convert the model
    if args.convert_model:
        model = models.convert_model(model, anchor_params=anchor_params)

    detections_raw = _get_detections(
        generator,
        model,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        save_path=args.save_path if args.save_output_imgs else None,
        inner_layer_name=args.inner_features_layer if get_inner_features else None
    )

    if get_inner_features:
        all_detections, all_features = detections_raw
        features_file_path = os.path.join(args.save_path, args.inner_features_file)
        np.savez_compressed(features_file_path, **{'x': all_features})
    else:
        all_detections = detections_raw

    filtered_detections_df = pd.DataFrame(columns=['frame_path', 'class', 'x1', 'y1', 'x2', 'y2', 'confidence'])
    class_map = {v: k for k, v in generator.classes.items()}
    for image_idx in range(generator.size()):
        frame_path = generator.image_path(image_idx)
        for label in range(generator.num_classes()):
            class_detections = all_detections[image_idx][label]
            if class_detections.size == 0:
                continue
            for detection_idx in range(class_detections.shape[0]):
                filtered_detections_df = filtered_detections_df.append({
                    'frame_path': frame_path,
                    'class': label,
                    'x1': class_detections[detection_idx, 0],
                    'y1': class_detections[detection_idx, 1],
                    'x2': class_detections[detection_idx, 2],
                    'y2': class_detections[detection_idx, 3],
                    'confidence': class_detections[detection_idx, 4]
                }, ignore_index=True)

    output_class_map_path = os.path.join(args.save_path, args.class_map_filename)
    with open(output_class_map_path, 'w') as jf:
        json.dump(class_map, jf)

    output_csv_path = os.path.join(args.save_path, args.predictions_filename)
    filtered_detections_df.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    main()
