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

"""Alternate between training and evaluation for each epoch.

To see various training arguments:
    python train.py --help
"""
import os

import tensorflow as tf

from gti.model.utils import (
    configure_model,
    get_vars_to_restore,
    train_epoch,
    eval_epoch,
    format_metrics,
    tag_checkpoint,
    model_factory
)
from gti.data_utils import load_tfrecords, load_image_files
from gti.param_parser import TrainParser 


def main(args):
    with tf.device("/cpu:0"):
        # Put data pipeline on CPU for performance, per:
        #   https://www.tensorflow.org/guide/performance/overview#preprocessing_on_the_cpu
        train_dataset = load_tfrecords(
            dir_path=args.train_data_dir,
            image_size=args.image_size,
            batch_size=args.train_batch_size,
            is_training=True
        )
        val_dataset = load_tfrecords(
            dir_path=args.val_data_dir,
            image_size=args.image_size,
            batch_size=args.val_batch_size,
        )
        # make reinitializable dataset iterator for train & validation
        iterator = tf.data.Iterator.from_structure(
            train_dataset.output_types,
            train_dataset.output_shapes
        )
        train_data_init_op = iterator.make_initializer(train_dataset)
        val_data_init_op = iterator.make_initializer(val_dataset)
        images, labels = iterator.get_next()

    # configure model specifications for train & validation
    model = model_factory(
        name=args.net,
        chip=args.chip,
        classes=args.classes,
        quant_w=args.quant_w,
        quant_act=args.quant_act,
        fuse=args.fuse,
        checkpoint=args.resume_from
    )
    train_spec = configure_model(
        model=model,
        is_training=True,
        inputs=images,
        labels=labels,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_gpus=args.num_gpus
    )
    val_spec = configure_model(
        model=model,
        is_training=False,
        inputs=images,
        labels=labels,
        weight_decay=args.weight_decay,
        reuse=True,
        num_gpus=args.num_gpus
    )
    
    # Create necessary directories if not already exist
    for i in [args.last_checkpoint_dir, args.best_checkpoint_dir]:
        if not os.path.exists(i):
            os.makedirs(i)

    with tf.Session() as sess:
        sess.run(train_spec["variables_init_op"])
        if args.resume_from is not None:
            begin_at_epoch = int(args.resume_from.split("-")[-1])
        else:
            begin_at_epoch = 0
        if not args.fuse:
            if args.resume_from is not None:
                print("Restoring checkpoint from {}".format(args.resume_from))
                print("Beginning at epoch {}".format(begin_at_epoch))
                vars_to_restore = get_vars_to_restore(args.resume_from)
                restorer = tf.train.Saver(var_list=vars_to_restore)
                restorer.restore(sess, args.resume_from)
            else:
                print("===============================================================================")
                print("WARNING: --resume_from checkpoint flag is not set. Training model from scratch.")
                print("===============================================================================")
        

        # Saver to handle lateset checkpoints
        last_saver = tf.train.Saver(max_to_keep=args.keep_last_n_checkpoints)

        # Save best checkpoint
        best_saver = tf.train.Saver(max_to_keep=1)
        best_val_accuracy = 0.0

        # Training loop
        for epoch in range(begin_at_epoch, begin_at_epoch + args.num_epochs):
            sess.run(train_data_init_op)
            print("Epoch {}/{}".format(epoch + 1, begin_at_epoch + args.num_epochs))
            train_metrics = train_epoch(
                session=sess,
                model_spec=train_spec,
                num_samples=args.train_size,
                batch_size=args.train_batch_size
            )
            print("Training " + format_metrics(train_metrics))

            sess.run(val_data_init_op)
            val_metrics = eval_epoch(
                session=sess,
                model_spec=val_spec,
                num_samples=args.val_size,
                batch_size=args.val_batch_size,
            )
            print("Validation " + format_metrics(val_metrics))

            # save latest checkpoint
            print("Saving latest checkpoint to {}".format(args.last_checkpoint_dir))
            checkpoint_tags = tag_checkpoint(
                chip=args.chip,
                quant_w=args.quant_w,
                quant_act=args.quant_act,
                fuse=args.fuse
            )
            last_saver.save(
                sess,
                os.path.join(args.last_checkpoint_dir, model.name + checkpoint_tags + "-epoch"),
                global_step=epoch + 1
            )

            # if checkpoint better than before, save to best
            if val_metrics["accuracy"] > best_val_accuracy:
                print("===========================================================================")
                print("Found better checkpoint! Saving to {}".format(args.best_checkpoint_dir))
                print("===========================================================================")
                best_saver.save(
                    sess,
                    os.path.join(args.best_checkpoint_dir, model.name + checkpoint_tags + "-epoch"),
                    global_step=epoch + 1
                )
                best_val_accuracy = val_metrics["accuracy"]
            print()

if __name__ == "__main__":
    parser = TrainParser()
    args = parser.parse_args()
    main(args)