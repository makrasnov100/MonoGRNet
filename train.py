#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the TensorDetect model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys
import cv2

# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'include')

import tensorvision.train as train
import tensorvision.utils as utils

flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('project', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('hypes', 'hypes/kittiBox.json',
                    'File storing model parameters.')

tf.app.flags.DEFINE_boolean(
    'save', True, ('Whether to save the run. In case --nosave (default) '
                   'output will be saved to the folder TV_DIR_RUNS/debug, '
                   'hence it will get overwritten by further runs.'))


def do_custom_eval_setup(hypes):
    full_input_dir = os.path.join(hypes['dirs']['data_dir'], hypes['data']['input_dir'])
    val_data_dir = os.path.join(hypes['dirs']['data_dir'], hypes['data']['val_dir'])
    # Check if path exists
    if not os.path.exists(full_input_dir):
        print(full_input_dir  + " does not exist")
        return
    
    # Get all images from the directory
    images = [f for f in os.listdir(full_input_dir) if os.path.isfile(os.path.join(full_input_dir, f))]
    print("Inference Images: ")
    print(images)
    val_file_lines = []
    for image in images:
        # Resize image keeping aspect ratio to have height max_height
        image_path = os.path.join(full_input_dir, image)
        output_path = os.path.join(val_data_dir, image) 
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        aspect_ratio = width / height
        target_height = hypes['resize']['target_height']
        max_height = hypes['resize']['max_height']
        max_width = hypes['resize']['max_width']
        vertical_border = (max_height - target_height) // 2
        new_width = int(target_height * aspect_ratio)
        img = cv2.resize(img, (new_width, target_height), interpolation = cv2.INTER_AREA)

        # Crop image to have width max_width or add black borders if width < max_width
        height, width = img.shape[:2]
        if width < max_width:
            border = (max_width - width) // 2
            img = cv2.copyMakeBorder(img, vertical_border, vertical_border, border, border, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif width > max_width:
            border = (width - max_width) // 2
            img = img[:, border:width - border]

        # Save image and return new path
        # - save image to new path
        cv2.imwrite(output_path, img)
        # - check by printing list of files in output directory
        print("Output Directory Files: ")
        print(os.listdir(val_data_dir))
        # - save a placeholder calib file
        calib_name = image.split('.')[0] + ".txt"
        calib_dir = os.path.join(hypes['dirs']['data_dir'], hypes['data']['calib_dir'])
        calib_path = os.path.join(calib_dir, calib_name)
        global_calib_path = os.path.join(hypes['dirs']['data_dir'], "global_config.txt")
        # make directory if it does not exist
        try:
            os.makedirs(calib_dir)
        except OSError:
            pass
        with open(calib_path, 'w') as f:
            with open(global_calib_path, 'r') as global_f:
                f.write(global_f.read())
        # - save a placeholder label file
        label_name = image.split('.')[0] + ".txt"
        label_dir = os.path.join(hypes['dirs']['data_dir'], hypes['data']['label_dir'])
        label_path = os.path.join(label_dir, label_name)
        try:
            os.makedirs(label_dir)
        except OSError:
            pass
        with open(label_path, 'w') as f:
            f.write("Car 1.00 0 2.52 0.00 222.42 211.82 374.00 1.52 1.54 3.68 -2.80 1.76 1.74 1.57")
            
        # - save a line in val.txt file
        custom_image_path = os.path.join(hypes['data']['custom_image_dir'], image)
        custom_label_path = os.path.join(hypes['data']['custom_label_dir'], label_name)
        val_file_lines.append(custom_image_path + " " + custom_label_path)

    # Write val.txt file
    val_file_path = os.path.join(hypes['dirs']['data_dir'], 'KittiBox', "val.txt")
    with open(val_file_path, 'w') as f:
        f.write('\n'.join(val_file_lines))
    train_file_path = os.path.join(hypes['dirs']['data_dir'], 'KittiBox', "train.txt")
    with open(train_file_path, 'w') as f:
        f.write('\n'.join(val_file_lines[:1]))

def main(_):
    utils.set_gpus_to_use()

    try:
        import tensorvision.train
    except ImportError:
        logging.error("Could not import the submodules.")
        logging.error("Please execute:"
                      "'git submodule update --init --recursive'")
        exit(1)

    with open(tf.app.flags.FLAGS.hypes, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)
    #utils.load_plugins()

    if 'TV_DIR_RUNS' in os.environ:
        os.environ['TV_DIR_RUNS'] = os.path.join(os.environ['TV_DIR_RUNS'],
                                                 'KittiBox')
    utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)

    utils._add_paths_to_sys(hypes)

    logging.info("Initialize training folder")
    train.initialize_training_folder(hypes)
    #train.maybe_download_and_extract(hypes)

    logging.info("Perform custom evaluation setup")
    do_custom_eval_setup(hypes)

    logging.info("Start training")
    train.do_training(hypes)


if __name__ == '__main__':
    tf.app.run()
