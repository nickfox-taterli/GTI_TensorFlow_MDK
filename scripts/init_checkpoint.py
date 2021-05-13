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

"""Create initial TensorFlow checkpoint from pretrained numpy weights.

This script is only needed when there's no GTI TensorFlow checkpoint, but there's a pretrained
numpy weights file. Numpy files are incompatible with the workflow, so they need to be first
converted to TensorFlow checkpoint.

Numpy weights can be ported from pretrained models available online:
e.g. VGG16:
    https://github.com/machrisaa/tensorflow-vgg
"""
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import tensorflow as tf

from gti.model.utils import configure_model, model_factory
from gti.param_parser import CheckpointInitParser


def main(args):
    if not os.path.exists(args.npy):
        raise FileNotFoundError("{}; No such file or directory".format(args.npy))
    images = tf.placeholder(shape=[1, args.image_size, args.image_size, 3], dtype=tf.float32)
    labels = tf.placeholder(shape=[1], dtype=tf.int64)
    model = model_factory(
        name=args.net,
        chip=args.chip,
        classes=args.classes,
        npy=args.npy
    )
    model_spec = configure_model(
        model=model,
        is_training=False,
        inputs=images,
        labels=labels
    )
    with tf.Session() as sess:
        print("Converting numpy weights file to checkpoint")
        sess.run(model_spec["variables_init_op"])
        if not os.path.exists(args.save_checkpoint_to):
            os.makedirs(args.save_checkpoint_to)
        print("Saving initial checkpoint to {}".format(args.save_checkpoint_to))
        last_saver = tf.train.Saver()
        last_saver.save(
            sess,
            os.path.join(args.save_checkpoint_to, model.name + "-epoch"),
            global_step=1
        )

if __name__ == "__main__":
    parser = CheckpointInitParser()
    args = parser.parse_args()
    main(args)
