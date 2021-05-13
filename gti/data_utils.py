'''
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

"""Handle data loading and preprocessing."""
import glob
import os
from random import shuffle

import numpy as np
import tensorflow as tf

import gti.chip.spec


IMAGENET_BGR_MEAN = [103.94, 116.78, 123.168]  # [B MEAN, G MEAN, R MEAN]
_UPSCALE = 0.875  # upscale image size for random cropping
_IMAGE_SIZE = None


# Assume GTI generated ImageNet TFRecords & preprocessing
def load_tfrecords(
        dir_path,
        image_size,
        batch_size=64,
        is_training=False,
        shuffle_buffer_size=10000,
        truncate_input=True
    ):
    """Load TFRecords for training and evaluation on PC."""
    _set_image_size(size=image_size)
    filenames = _get_tfrecord_filenames(dir_path, is_training)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_preprocess_proto, num_parallel_calls=os.cpu_count())
    if is_training:  # for training, augment and shuffle
        dataset = dataset.map(_augment, num_parallel_calls=os.cpu_count())
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    else:  # for evaluation, center crop or pad
        dataset = dataset.map(_resize_and_center_crop, num_parallel_calls=os.cpu_count())
    if truncate_input:
        dataset = dataset.map(_simulate_chip_input_truncation, num_parallel_calls=os.cpu_count())
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

def load_tfrecords_for_chip(dir_path, image_size):
    """Load TFRecords for chip evaluation/inference."""
    _set_image_size(size=image_size)
    filenames = _get_tfrecord_filenames(dir_path, is_training=False)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_preprocess_proto, num_parallel_calls=os.cpu_count())
    dataset = dataset.map(_resize_and_center_crop, num_parallel_calls=os.cpu_count())
    dataset = dataset.map(_to_uint8, num_parallel_calls=os.cpu_count())
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(1)
    return dataset

def read_image_for_chip(filename, image_size):
    """Load a single image file for chip evaluation/inference."""
    _set_image_size(size=image_size)
    image = tf.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    red, green, blue = tf.split(image, num_or_size_splits=3, axis=2)
    image = tf.concat([blue, green, red], axis=2)
    image, _ = _resize_and_center_crop(image, None)
    image, _ = _to_uint8(image, None)
    return image


# Assume GTI Face5 dataset, jpg images.
def load_image_files(
        dir_path,
        image_size,
        batch_size=64,
        is_training=False,
        shuffle_buffer_size=10000,
        truncate_input=True
    ):
    """Load image files for training and evaluation on PC."""
    _set_image_size(size=image_size)
    files, labels = _get_files_and_labels(dir_path)
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(files), tf.convert_to_tensor(labels))
    )
    dataset = dataset.map(_preprocess_image_file, num_parallel_calls=os.cpu_count())
    if is_training:
        dataset = dataset.map(_augment_face5, num_parallel_calls=os.cpu_count())
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(_resize, num_parallel_calls=os.cpu_count()) 
    if truncate_input:
        dataset = dataset.map(_simulate_chip_input_truncation, num_parallel_calls=os.cpu_count())
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

def load_image_files_for_chip(dir_path, image_size):
    """Load image files for chip evaluation/inference."""
    _set_image_size(size=image_size)
    files, labels = _get_files_and_labels(dir_path)
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(files), tf.convert_to_tensor(labels))
    )
    dataset = dataset.map(_preprocess_image_file, num_parallel_calls=os.cpu_count())
    dataset = dataset.map(_resize, num_parallel_calls=os.cpu_count())
    dataset = dataset.map(_to_uint8, num_parallel_calls=os.cpu_count())
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(1)
    return dataset


def _get_tfrecord_filenames(dir_path, is_training):
    if not os.path.exists(dir_path):
        raise FileNotFoundError("{}; No such file or directory.".format(dir_path))
    filenames = sorted(glob.glob(os.path.join(dir_path, "*.tfrecord")))
    if not filenames:
        raise FileNotFoundError("No TFRecords found in {}".format(dir_path))
    if is_training:
        shuffle(filenames)
    return filenames


def _preprocess_proto(example_proto):
    """Read image from protocol buffer and convert to BGR by GTI convention."""
    encoding_scheme = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
    }
    proto = tf.parse_single_example(example_proto, encoding_scheme)
    image = tf.image.decode_png(proto["image/encoded"], channels=3)
    red, green, blue = tf.split(image, num_or_size_splits=3, axis=2)
    image = tf.concat([blue, green, red], axis=2)
    label = proto["image/class/label"]
    return image, label


def _simulate_chip_input_truncation(image, label):
    # NOTE: input is truncated to 5 bits on chip, i.e. [0, 255] --> [0, 31]
    # Simulate input truncation ((x >> 2) + 1) >> 1
    image = tf.cast(image, tf.uint8)
    image = tf.bitwise.right_shift(image, tf.ones_like(image) * 2)
    image = image + 1
    image = tf.bitwise.right_shift(image, tf.ones_like(image))
    image = tf.cast(image, tf.float32)
    image = tf.clip_by_value(image, 0, gti.chip.spec.get_max_activation()) 
    return image, label


def _augment(image, label):
    """Augment ImageNet"""
    image = tf.image.resize_images(image, [int(_IMAGE_SIZE / _UPSCALE), int(_IMAGE_SIZE / _UPSCALE)])
    image = tf.image.random_crop(image, [_IMAGE_SIZE, _IMAGE_SIZE, 3])
    image = tf.image.random_flip_left_right(image)
    return image, label


def _resize_and_center_crop(image, label):
    """Resize and center crop image.

    New size = image size / upscale, then center crop to image size
    """
    image = tf.image.resize_images(image, [int(_IMAGE_SIZE / _UPSCALE), int(_IMAGE_SIZE / _UPSCALE)])
    image = tf.image.central_crop(image, central_fraction=_UPSCALE)
    return image, label


def _subtract_mean(image, label):
    image = tf.cast(image, tf.float32)
    image -= IMAGENET_BGR_MEAN
    return image, label


def _resize(image, label):
    """Resize to image_size x image_size for evaluation/inference."""
    image = tf.image.resize_images(image, [_IMAGE_SIZE, _IMAGE_SIZE])
    return image, label


def _augment_face5(image, label):
    """Augment Face 5 dataset of 5 celebrity faces."""
    image = tf.image.random_flip_left_right(image) 
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_jpeg_quality(image, 80, 100)
    if np.random.uniform() > 0.5:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
    if np.random.uniform() > 0.5:
        image = tf.image.resize_images(image, [int(_IMAGE_SIZE / _UPSCALE), int(_IMAGE_SIZE / _UPSCALE)])
        image = tf.image.random_crop(image, [_IMAGE_SIZE, _IMAGE_SIZE, 3])
    return image, label


def _get_files_and_labels(dir_path, extension="jpg"):
    """ Get list of filenames and list of labels.

    Assume files are organized in such way:
        dir_path/class subfolder/file.extension
        dir_path/another class subfolder/another file.extension
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError("{}; No such file or directory.".format(dir_path))
    subfolders = sorted(os.listdir(dir_path))
    files = []
    labels = []
    for idx, subfolder in enumerate(subfolders):
        subfolder_files = glob.glob(os.path.join(dir_path, subfolder, "*." + extension))
        files += subfolder_files
        try:  # if class name is a number, try converting it to integer label directly
            labels += [int(subfolder)] * len(subfolder_files)
        except ValueError:
            # if class name is not convertible to integer label, use its index as integer label 
            labels += [idx] * len(subfolder_files)
    if not files or not labels:
        raise FileNotFoundError(
            "{} does not contain files with extension \"{}\"".format(dir_path, extension)
        )
    return files, labels


def _preprocess_image_file(image_filename, label):
    image = tf.read_file(image_filename) 
    image = tf.image.decode_jpeg(image, channels=3)
    red, green, blue = tf.split(image, num_or_size_splits=3, axis=2)
    image = tf.concat([blue, green, red], axis=2)
    return image, label


def _set_image_size(size):
    "Set preprocessing image size. NOTE: function creates side-effect."
    if size not in gti.chip.spec.ALLOWED_IMAGE_SIZES:
        raise ValueError("Incompatible input image size")
    global _IMAGE_SIZE
    _IMAGE_SIZE = size


def _to_uint8(image, label):
    return tf.cast(image, tf.uint8), label