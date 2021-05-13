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

"""Example script to perform chip inference."""
import glob
import os
import time

import tensorflow as tf

from gti.chip.driver import GtiModel
from gti.data_utils import (
    load_tfrecords_for_chip,
    load_image_files_for_chip,
    read_image_for_chip
)
from gti.param_parser import ChipInferParser
from gti.model.utils import get_vars_to_restore, model_factory, path_helper


def main(args):
    predict_image(args)
    evaluate_dataset(args)


def predict_image(args):
    """Example: single image inference on chip using full model (convolutional layers + fully
    connected layers).
    """
    model = model_factory(
        name=args.net,
        chip=args.chip,
        classes=args.classes,
        checkpoint=args.checkpoint
    ) 
    with tf.variable_scope(model.name, reuse=tf.AUTO_REUSE):
        chip_output = tf.placeholder(tf.float32, shape=model.chip_output_shape)
        logits = model.build_host_net(chip_output, is_training=False)
        predictions = tf.argmax(logits, axis=1)

    with tf.Session() as sess:
        model_path = path_helper(net=args.net, chip=args.chip).model
        chip_model = GtiModel(model_path)
        print("Example: chip model inference on single images")
        samples_dir = os.path.join("data", "samples", "imagenet")
        image_extension = "JPEG"
        search_pattern = samples_dir + "/**/*." + image_extension 
        image_filenames = glob.glob(search_pattern, recursive=True)

        # Load checkpoint for on-host layers
        sess.run(tf.global_variables_initializer())
        vars_to_restore = get_vars_to_restore(args.checkpoint)
        restorer = tf.train.Saver(var_list=vars_to_restore)
        restorer.restore(sess, args.checkpoint)
        reader = tf.train.NewCheckpointReader(args.checkpoint)
        last_relu_cap = reader.get_tensor(args.last_relu_cap)[0]
        print("Last ReLU cap: {}".format(last_relu_cap))
        if args.ten_bit_act:
            act_bits = 10
        else:
            act_bits = 5
        for image_filename in image_filenames: 
            numpy_array = sess.run(read_image_for_chip(image_filename, image_size=args.image_size))
            chip_out = chip_model.evaluate(
                numpy_array=numpy_array,
                last_relu_cap=last_relu_cap,
                activation_bits=act_bits
            )
            pred = sess.run(predictions, feed_dict={chip_output: chip_out})
            print("Image file: {}".format(image_filename))
            print("Class prediction: {}".format(pred))
            print()
        chip_model.release()
        print("-------------------------------------------------------------------------------")


def evaluate_dataset(args):
    """Example: evaluate on dataset on chip using full model (convolutional layers + fully connected
    layers).
    """
    model = model_factory(
        name=args.net,
        chip=args.chip,
        classes=args.classes,
        checkpoint=args.checkpoint
    ) 
    with tf.variable_scope(model.name, reuse=tf.AUTO_REUSE):
        chip_output = tf.placeholder(tf.float32, shape=model.chip_output_shape)
        logits = model.build_host_net(chip_output, is_training=False)
        predictions = tf.argmax(logits, axis=1)
    
    with tf.device("/cpu:0"):
        dataset = load_tfrecords_for_chip(
            dir_path=args.data_dir,
            image_size=args.image_size
        )
        iterator = dataset.make_initializable_iterator()
        itr_image, itr_label = iterator.get_next()
        data_init_op = iterator.initializer
    
    with tf.Session() as sess:
        sess.run(data_init_op)
        sess.run(tf.global_variables_initializer())

        # Load checkpoint for on-host layers
        vars_to_restore = get_vars_to_restore(args.checkpoint)
        restorer = tf.train.Saver(var_list=vars_to_restore)
        restorer.restore(sess, args.checkpoint)
        reader = tf.train.NewCheckpointReader(args.checkpoint)
        last_relu_cap = reader.get_tensor(args.last_relu_cap)[0]
        print("Last ReLU cap: {}".format(last_relu_cap))
        print("Evaluating chip model on dataset of {}".format(args.data_size))
        model_path = path_helper(net=args.net, chip=args.chip).model
        chip_model = GtiModel(model_path)
        is_correct = 0
        if args.ten_bit_act:
            act_bits = 10
        else:
            act_bits = 5
        for i in range(args.data_size):
            image, label = sess.run([itr_image, itr_label])
            chip_out = chip_model.evaluate(
                numpy_array=image,
                last_relu_cap=last_relu_cap,
                activation_bits=act_bits
            )
            pred = sess.run(predictions, feed_dict={chip_output: chip_out})[0]
            if pred == int(label[0]):
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
    parser = ChipInferParser() 
    args = parser.parse_args()
    main(args)