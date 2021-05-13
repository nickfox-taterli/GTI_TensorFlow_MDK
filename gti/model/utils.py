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

"""Model helper functions."""
from collections import namedtuple
import importlib
import os

import tensorflow as tf
from tensorflow.python.client import device_lib
import tqdm


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def tower_outputs(model, is_training, inputs, labels, weight_decay, reuse):
    with tf.variable_scope(model.name, reuse=reuse):
        logits = model.build(
            inputs=inputs,
            is_training=is_training
        )
        predictions = tf.argmax(logits, axis=1)
    # define loss function, by default add L2 regularization
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    l2_loss = weight_decay * tf.add_n(
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()]
    )
    loss = cross_entropy + l2_loss
    return loss, logits, predictions


def configure_model(
        model,
        is_training,
        inputs,
        labels,
        learning_rate=1e-4,
        weight_decay=1e-4,
        reuse=tf.AUTO_REUSE,
        num_gpus=1
    ):
    """Configure model for training & evaluation."""
    losses = []
    logits = []
    predictions = []
    batch_size = tf.shape(inputs)[0]
    devices = get_available_devices(num_request_gpus=num_gpus)
    num_devices = len(devices)
    remainder = batch_size % num_devices 
    if remainder == 0:  # batch size divides evenly by number of devices, split batch by devices 
        split_inputs = tf.split(inputs, num_devices)
        split_labels = tf.split(labels, num_devices)
    else:  # batch divides unevenly by number of devices, concat the remainder samples to the last split
        split_inputs = tf.split(inputs[:batch_size-remainder], num_devices)
        split_inputs[-1] = tf.concat([split_inputs[-1], inputs[batch_size-remainder:]], 0)
        split_labels = tf.split(labels[:batch_size-remainder], num_devices)
        split_labels[-1] = tf.concat([split_labels[-1], labels[batch_size-remainder:]], 0)

    if is_training:
        grads = []
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        global_step = tf.train.get_or_create_global_step()
        for i, device in enumerate(devices):
            with tf.device(device):
                tower_loss, tower_logits, tower_predictions = tower_outputs(
                    model=model,
                    is_training=is_training,
                    inputs=split_inputs[i],
                    labels=split_labels[i],
                    weight_decay=weight_decay,
                    reuse=reuse
                )
                losses.append(tower_loss)
                tower_grads = optimizer.compute_gradients(tower_loss)
                grads.append(tower_grads)
                logits.append(tower_logits)
                predictions.append(tower_predictions)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            grads = average_gradients(grads)
            apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
            train_op = tf.group(apply_gradient_op)
            loss = tf.reduce_mean(losses)
            logits = tf.concat(logits, 0)
            predictions = tf.concat(predictions, 0)
    else:
        for i, device in enumerate(devices):
            with tf.device(device):
                tower_loss, tower_logits, tower_predictions = tower_outputs(
                    model=model,
                    is_training=is_training,
                    inputs=split_inputs[i],
                    labels=split_labels[i],
                    weight_decay=weight_decay,
                    reuse=reuse
                )
                losses.append(tower_loss)
                logits.append(tower_logits)
                predictions.append(tower_predictions)
        loss = tf.reduce_mean(losses)
        logits = tf.concat(logits, 0)
        predictions = tf.concat(predictions, 0)

    # Metrics
    with tf.variable_scope("metrics"):
        metrics = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions),
            "loss": tf.metrics.mean(loss)
        }
    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Init and reset tf.metrics variables
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Model specifications
    model_spec = {}
    model_spec["variables_init_op"] = tf.global_variables_initializer()
    model_spec["metrics_init_op"] = metrics_init_op
    model_spec["metrics"] = metrics
    model_spec["update_metrics"] = update_metrics_op
    model_spec["logits"] = logits
    model_spec["predictions"] = predictions

    if is_training:
        model_spec["train_op"] = train_op

    return model_spec


def get_vars_to_restore(checkpoint_path):
    """Get list of variables to restore from checkpoint."""
    reader = tf.train.NewCheckpointReader(checkpoint_path)
    # map name to shape in checkpoint variables 
    checkpoint_vars = reader.get_variable_to_shape_map()
    # map name to variable in current graph variables 
    graph_vars = {v.name.split(":")[0]: v for v in tf.global_variables()}
    # intersection of the two sets of variables
    intersection = set(graph_vars.keys()).intersection(set(checkpoint_vars.keys()))
    vars_to_restore = []
    with tf.variable_scope('', reuse=True):
        for i in intersection:
            if graph_vars[i].get_shape().as_list() != checkpoint_vars[i]:
                raise Exception(
                    "Found variable {} with shape conflict, in graph {}, in checkpoint {}"
                    .format(i, graph_vars[i].get_shape().as_list(), checkpoint_vars[i])
                )
            vars_to_restore.append(graph_vars[i])
    return vars_to_restore


def train_epoch(session, model_spec, num_samples, batch_size):
    train_op = model_spec["train_op"]
    metrics = model_spec["metrics"]
    update_metrics = model_spec["update_metrics"]
    session.run(model_spec["metrics_init_op"])
    num_steps = _calc_num_steps(num_samples, batch_size)
    metric_values = {k: v[0] for k, v in metrics.items()}
    all_ops = {"train": train_op, "update": update_metrics}
    for k, v in metric_values.items():
        all_ops[k] = v 
    with tqdm.trange(num_steps) as t:
        for _ in t:
            step_results = session.run(all_ops)
            stats = {k: v for k, v in step_results.items() if k not in {"train", "update"}}
            t.set_postfix(stats)
    return session.run(metric_values)


def eval_epoch(session, model_spec, num_samples, batch_size):
    metrics = model_spec["metrics"]
    update_metrics = model_spec["update_metrics"]
    session.run(model_spec["metrics_init_op"])
    num_steps = _calc_num_steps(num_samples, batch_size)
    metric_values = {k: v[0] for k, v in metrics.items()}
    all_ops = {"update": update_metrics}
    for k, v in metric_values.items():
        all_ops[k] = v
    with tqdm.trange(num_steps) as t:
        for _ in t:
            step_results = session.run(all_ops)
            stats = {k: v for k, v in step_results.items() if k not in {"update"}}
            t.set_postfix(stats)
    return session.run(metric_values)


def format_metrics(metrics, sep="; "):
    return sep.join("{}: {:.6f}".format(k, metrics[k]) for k in sorted(metrics.keys()))


def _calc_num_steps(num_samples, batch_size):
    return (num_samples + batch_size - 1) // batch_size


def tag_checkpoint(chip, quant_w, quant_act, fuse):
    """Tag checkpoint to indicate quantization schemes & chip"""
    tags = "-" + chip 
    if quant_w:
        tags += "-qw"
    if quant_act:
        tags += "-qa"
    if fuse:
        tags += "-fs"
    return tags


def model_factory(
        name,
        chip,
        classes=None,
        quant_w=False,
        quant_act=False,
        fuse=False,
        checkpoint=None,
        npy=None
    ):
    module = importlib.import_module("gti.model." + name)
    model_class = getattr(module, name)
    model_instance = model_class(
        target_chip=chip,
        classes=classes,
        quant_w=quant_w,
        quant_act=quant_act,
        fuse=fuse,
        checkpoint=checkpoint,
        npy=npy
    )
    return model_instance
    

TaskPath = namedtuple("TaskPath", ["dat_json", "model_json", "model"])
def path_helper(net, chip):
    """Helper function to get default paths to files associated with conversion and inference.
    
    Args:
        net (str): network name
        chip (str): GTI chip series
    
    Returns:
        task path (namedtuple): paths to associated files
    """
    path_prefix = "_".join([chip, net])
    save_dir = "nets"
    return TaskPath(
        dat_json=os.path.join(save_dir, path_prefix + "_dat.json"),
        model_json=os.path.join(save_dir, path_prefix + "_model.json"),
        model=os.path.join(save_dir, path_prefix + ".model")
    )


def get_available_devices(num_request_gpus):
    if num_request_gpus < 0:
        raise ValueError("Number of GPUs to use must be >= 0, where 0 means CPU-only")
    available_gpus = [i.name for i in device_lib.list_local_devices() if i.device_type == "GPU"] 
    num_available_gpus = len(available_gpus)
    if not available_gpus or num_request_gpus < 1:  # no GPU or requested 0 GPU
        return ["/device:CPU:0"]
    if num_request_gpus < num_available_gpus:
        # user requested fewer than available GPUs, use up to requested number
        return available_gpus[:num_request_gpus] 
    return available_gpus   # all other cases default to use all available GPUs
