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

"""Evaluate a checkpoint on validation set.

See a list of options by typing:
    python eval.py --help
"""
import os

import tensorflow as tf

from gti.model.utils import configure_model, eval_epoch, format_metrics, model_factory
from gti.data_utils import load_tfrecords, load_image_files
from gti.param_parser import EvalParser


def main(args):
    with tf.device("/cpu:0"):
        dataset = load_tfrecords(
            dir_path=args.data_dir,
            image_size=args.image_size,
            batch_size=args.batch_size
        )
        iterator = dataset.make_initializable_iterator()
        data_init_op = iterator.initializer
        images, labels = iterator.get_next()
    model = model_factory(
        name=args.net,
        chip=args.chip,
        classes=args.classes,
        quant_w=args.quant_w,
        quant_act=args.quant_act,
        fuse=args.fuse,
        checkpoint=args.checkpoint
    )
    model_spec = configure_model(
        model=model,
        is_training=False,
        inputs=images,
        labels=labels,
        num_gpus=args.num_gpus
    )
    with tf.Session() as sess:
        sess.run(data_init_op)
        sess.run(model_spec["variables_init_op"])
        if not args.fuse:  # use TF default restorer if not fuse batch norm and ReLU cap
            restorer = tf.train.Saver()
            print("Restoring checkpoint from {}".format(args.checkpoint))
            restorer.restore(sess, args.checkpoint)
        print("Running evaluation")
        eval_metrics = eval_epoch(
            session=sess,
            model_spec=model_spec,
            num_samples=args.data_size,
            batch_size=args.batch_size
        )
        print(format_metrics(eval_metrics))

if __name__ == "__main__":
    parser = EvalParser()
    args = parser.parse_args()
    main(args)