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

"""Evaluate ReLU output ranges, and use to initialize quantized ReLU activations.

See a list of options by typing:
    python set_relu_caps.py --help
"""
import json
import os

import numpy as np
import tensorflow as tf
import tqdm

from gti.data_utils import load_tfrecords, load_image_files
from gti.param_parser import ReLUCapParser 
from gti.model.utils import model_factory


def main(args):
    with tf.device("/cpu:0"):
        dataset = load_tfrecords(
            dir_path=args.data_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            is_training=True
        )
        iterator = dataset.make_initializable_iterator()
        data_init_op = iterator.initializer
        images, _ = iterator.get_next()
    model = model_factory(
        name=args.net,
        chip=args.chip,
        classes=args.classes,
        quant_w=args.quant_w,
        quant_act=args.quant_act
    )
    with tf.variable_scope(model.name, reuse=tf.AUTO_REUSE):
        _ = model.build(
            inputs=images,
            is_training=False
        )

    percentile_outputs = {}
    for k, v in model.relu_outputs.items():
        if v is None:
            raise Exception("ReLU output is not referenced for layer scope: {}".format(k))
        percentile_outputs[k] = tf.contrib.distributions.percentile(v, q=args.percentile)

    with tf.Session() as sess:
        sess.run(data_init_op)
        sess.run(tf.global_variables_initializer())
        restorer = tf.train.Saver()
        print("Restoring checkpoint from {}".format(args.checkpoint))
        restorer.restore(sess, args.checkpoint)
        print(
            "Evaluating activation outputs at {} percentile for {} batches"
            .format(args.percentile, args.num_batches)
        )

        relu_caps = {k: 0 for k in percentile_outputs.keys()}
        for _ in tqdm.trange(args.num_batches):
            batch_outputs = sess.run(percentile_outputs)
            relu_caps = {k: np.maximum(relu_caps[k], batch_outputs[k]) for k in relu_caps.keys()}
        print("Initial ReLU caps:")
        for k in sorted(relu_caps.keys()):
            print("\"{}\": {}".format(k, relu_caps[k]))
        
        # Overwrite model settings with computed ReLU caps
        print("Overwriting model settings file: {}".format(model.settings_path))
        with open(model.settings_path, "r+") as f:
            settings = json.load(f)
            settings["relu_cap"] = relu_caps
            f.seek(0)
            json.dump(settings, f, indent=4)
            f.truncate()


if __name__ == "__main__":
    parser = ReLUCapParser()
    args = parser.parse_args()
    main(args)
