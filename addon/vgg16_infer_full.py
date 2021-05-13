''''
All Rights Reserved.

Copyright (c) 2017-2019, Gyrfalcon technology Inc.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES;LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

"""Chip inference with full .model, including host layers."""
import argparse
import glob
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import tensorflow as tf

import gti.chip.spec
from gti.chip.driver import GtiModel
from gti.data_utils import load_tfrecords_for_chip, read_image_for_chip


def main(args):
    predict_image(args)
    evaluate_dataset(args)


def predict_image(args):
    """Example: single image inference on chip using full model (convolutional layers + fully
    connected layers).
    """
    with tf.Session() as sess:
        chip_model = GtiModel(args.model)
        print("Example: chip model inference on single images")
        samples_dir = os.path.join("data", "samples", "imagenet")
        image_extension = "JPEG"
        search_pattern = samples_dir + "/**/*." + image_extension 
        image_filenames = glob.glob(search_pattern, recursive=True)
        for image_filename in image_filenames: 
            numpy_array = sess.run(read_image_for_chip(image_filename, image_size=args.image_size))
            result = chip_model.full_inference(numpy_array=numpy_array)
            print("Image file: {}".format(image_filename))
            print("Class prediction: {}".format(result["result"][0]["index"]))
            print()
        chip_model.release()
        print("-------------------------------------------------------------------------------")


def evaluate_dataset(args):
    """Example: evaluate on dataset on chip using full model (convolutional layers + fully connected
    layers).
    """
    with tf.device("/cpu:0"):
        dataset = load_tfrecords_for_chip(
            dir_path=args.data_dir,
            image_size=gti.chip.spec.DEFAULT_IMAGE_SIZE
        )
        iterator = dataset.make_initializable_iterator()
        itr_image, itr_label = iterator.get_next()
        data_init_op = iterator.initializer
    
    with tf.Session() as sess:
        sess.run(data_init_op)
        print("Evaluating chip model on dataset of {}".format(args.data_size))
        is_correct = 0
        chip_model = GtiModel(args.model)
        for i in range(args.data_size):
            image, label = sess.run([itr_image, itr_label])
            result = chip_model.full_inference(numpy_array=image)
            if int(result["result"][0]["index"]) == int(label[0]):
                is_correct += 1
            progress_str = "evaluated: {}/{}; correct: {}/{}; accuracy: {}".format(
                i + 1,
                args.data_size,
                is_correct,
                i + 1,
                is_correct / (i + 1)
            )
            print(progress_str, end="\r")
        print(progress_str)
        chip_model.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=".model file path"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/imagenet/val",
        help="dataset directory for evaluation on chip"
    )
    parser.add_argument(
        "--data_size",
        type=int,
        default=50000,
        help="dataset size" 
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        choices=gti.chip.spec.ALLOWED_IMAGE_SIZES,
        help="image size == image height == image width" 
    )
    args = parser.parse_args()
    main(args)