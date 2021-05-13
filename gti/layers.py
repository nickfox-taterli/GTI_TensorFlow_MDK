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

"""GTI specific layers implementations."""
from collections import namedtuple
import logging
import os

import numpy as np
import tensorflow as tf

import gti.chip.spec
import gti.quantize


_DEBUG_LAYERS = os.environ.get("GTI_DEBUG_LAYERS") == "True" 
logging.basicConfig()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG if _DEBUG_LAYERS else logging.INFO)

# Do not clip convolution bias by default. If max|B| >> max|W|, might degrade on-chip performance,
# Try set to True and fine-tune model if that's the case.
_CLIP_CONVOLUTION_BIAS = os.environ.get("GTI_CLIP_BIAS") == "True"
class ClipBiasToWeight(tf.keras.constraints.Constraint):
    """Constraint on convolutional layer bias terms.

    Limit growth of biases to the maximum magnitdue of weights. Rationale: If bias grows too large,
    i.e. bias >> weight, likely the model has overfitted with biases, which may result in large
    quantization errors on chip.
    """
    def __init__(self, filter_):
        self.max_filter_magnitude = tf.reduce_max(tf.abs(filter_))

    def __call__(self, w):
        return tf.clip_by_value(w, -self.max_filter_magnitude, self.max_filter_magnitude)


def relu(
        inputs,
        layer_scope,
        quantize=False,
        cap=gti.chip.spec.get_max_activation(),
        activation_bits=5,
        trainable=False
    ):
    with tf.variable_scope(layer_scope):
        if quantize:
            return gti.quantize.quantized_relu(
                inputs=inputs,
                cap=cap,
                activation_bits=activation_bits,
                trainable=trainable
            )
        return tf.nn.relu(inputs)


def conv2d(
        inputs,
        in_channels,
        out_channels,
        layer_scope,
        use_bias=True,
        quantize=False,
        target_chip=None,
        mask_bitwidth=None,
        filter_initializer=None,
        bias_initializer=None,
        filter_size=(3, 3),
        stride=1,
    ):
    if filter_size not in gti.chip.spec.ALLOWED_CONVOLUTION_FILTER_SIZES:
        raise ValueError("Filter size not supported")
    if stride not in gti.chip.spec.ALLOWED_CONVOLUTION_STRIDES:
        raise ValueError("Convolution stride not supported")
    with tf.variable_scope(layer_scope):
        inputs, padding = _pad_like_chip(
            inputs=inputs,
            stride=stride,
            filter_size=filter_size
        )
        filter_height, filter_width = filter_size
        filter_shape = [filter_height, filter_width, in_channels, out_channels]
        if filter_initializer is None:
            filter_initializer = tf.glorot_uniform_initializer
        else:
            if not callable(filter_initializer):
                filter_initializer = tf.convert_to_tensor(filter_initializer, tf.float32)
                filter_shape = None
        filters = tf.get_variable(
            name="filters",
            shape=filter_shape,
            initializer=filter_initializer,
            trainable=True
        )
        strides = [1, stride, stride, 1]
        if not use_bias:
            if not quantize:
                return tf.nn.conv2d(inputs, filters, strides, padding) 
            else:
                return gti.quantize.quantized_conv2d(
                    inputs=inputs,
                    filters=filters,
                    biases=None,
                    target_chip=target_chip,
                    mask_bitwidth=mask_bitwidth,
                    strides=strides,
                    padding=padding
                ) 
        bias_shape = [out_channels]
        if bias_initializer is None:
            bias_initializer = tf.zeros_initializer
        else:
            if not callable(bias_initializer):
                bias_initializer = tf.convert_to_tensor(bias_initializer, tf.float32)
                bias_shape = None
        if _CLIP_CONVOLUTION_BIAS:
            bias_constraint = ClipBiasToWeight(filters)
        else:
            bias_constraint = None
        biases = tf.get_variable(
            name="biases",
            shape=bias_shape,
            initializer=bias_initializer,
            constraint=bias_constraint,
            trainable=True
        )
        if not quantize:
            return tf.nn.bias_add(tf.nn.conv2d(inputs, filters, strides, padding), biases)
        else:
            return gti.quantize.quantized_conv2d(
                inputs=inputs,
                filters=filters,
                biases=biases,
                target_chip=target_chip,
                mask_bitwidth=mask_bitwidth,
                strides=strides,
                padding=padding
            )

def deconv2d(
        inputs,
        in_channels,
        out_channels,
        layer_scope,
        upsampling_fill_mode,
        use_bias=True,
        quantize=False,
        target_chip=None,
        mask_bitwidth=None,
        filter_initializer=None,
        bias_initializer=None,
        filter_size=(3, 3),
        stride=1,
    ):
    """GTI device supported 'deconvolution', i.e. upsampling followed by GTI conv2d
    
    upsampling fill mode, see gti.chip.spec.UpSamplingFillMode:
        - REPEAT: fill with repeats of current value, for example:
            1 becomes [1, 1]
                      [1, 1]
        - ZERO: fill with zeros, for example:
            1 becomes [1, 0]
                      [0, 0]
    """
    if upsampling_fill_mode == gti.chip.spec.UpSamplingFillMode.REPEAT:
        out = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(inputs)
    elif upsampling_fill_mode == gti.chip.spec.UpSamplingFillMode.ZERO:
        out = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(inputs)
        overlay = tf.constant([[1, 0], [0, 0]], dtype=tf.float32)
        overlay = tf.tile(overlay, [tf.shape(out)[1] / 2, tf.shape(out)[2] / 2])
        overlay = tf.reshape(overlay, shape=[1, tf.shape(overlay)[0], tf.shape(overlay)[1], 1])
        out *= overlay
    else:
        raise ValueError("Invalid upsampling_fill_mode")
    return conv2d(
        inputs=out,
        in_channels=in_channels,
        out_channels=out_channels,
        layer_scope=layer_scope,
        use_bias=use_bias,
        quantize=quantize,
        target_chip=target_chip,
        mask_bitwidth=mask_bitwidth,
        filter_initializer=filter_initializer,
        bias_initializer=bias_initializer,
        filter_size=filter_size,
        stride=stride 
    )

def depthwise_conv2d(
        inputs,
        in_channels,
        stride,
        layer_scope,
        use_bias=False,
        quantize=False,
        target_chip=None,
        mask_bitwidth=None,
        filter_initializer=None,
        bias_initializer=None,
    ):
    if stride not in gti.chip.spec.ALLOWED_CONVOLUTION_STRIDES:
        raise ValueError("Stride not supported")
    with tf.variable_scope(layer_scope):
        inputs, padding = _pad_like_chip(
            inputs=inputs,
            stride=stride,
            filter_size=gti.chip.spec.DEFAULT_CONVOLUTION_FILTER_SIZE
        )
        filter_height, filter_width = gti.chip.spec.DEFAULT_CONVOLUTION_FILTER_SIZE
        filter_shape = [filter_height, filter_width, in_channels, 1]
        if filter_initializer is None:
            filter_initializer = tf.glorot_uniform_initializer
        else:
            if not callable(filter_initializer):
                filter_initializer = tf.convert_to_tensor(filter_initializer, tf.float32)
                filter_shape = None
        filters = tf.get_variable(
            name="filters",
            shape=filter_shape,
            initializer=filter_initializer,
            trainable=True
        )
        strides = [1, stride, stride, 1]
        if not use_bias:
            if not quantize:
                return tf.nn.depthwise_conv2d(inputs, filters, strides, padding) 
            else:
                return gti.quantize.quantized_depthwise_conv2d(
                    inputs=inputs,
                    filters=filters,
                    biases=None,
                    target_chip=target_chip,
                    mask_bitwidth=mask_bitwidth,
                    strides=strides,
                    padding=padding
                )     
        bias_shape = [in_channels]
        if bias_initializer is None:
            bias_initializer = tf.zeros_initializer
        else:
            if not callable(bias_initializer):
                bias_initializer = tf.convert_to_tensor(bias_initializer, tf.float32)
                bias_shape = None
        if _CLIP_CONVOLUTION_BIAS:
            bias_constraint = ClipBiasToWeight(filters)
        else:
            bias_constraint = None
        biases = tf.get_variable(
            name="biases",
            shape=bias_shape,
            initializer=bias_initializer,
            constraint=bias_constraint,
            trainable=True
        )
        if not quantize:
            return tf.nn.bias_add(
                tf.nn.depthwise_conv2d(inputs, filters, strides, padding),
                biases
            )
        else:
            return gti.quantize.quantized_depthwise_conv2d(
                inputs=inputs,
                filters=filters,
                biases=biases,
                target_chip=target_chip,
                mask_bitwidth=mask_bitwidth,
                strides=strides,
                padding=padding
            )


def fully_connected(
        inputs,
        in_size,
        out_size,
        layer_scope,
        weight_initializer=None,
        bias_initializer=None
    ):
    with tf.variable_scope(layer_scope):
        weight_shape = [in_size, out_size]
        if weight_initializer is None:
            weight_initializer = tf.glorot_uniform_initializer
        else:
            if not callable(weight_initializer):
                weight_initializer = tf.convert_to_tensor(weight_initializer, tf.float32)
                weight_shape = None
        weights = tf.get_variable(
            name="weights",
            shape=weight_shape,
            initializer=weight_initializer,
            trainable=True
        )
        bias_shape = [out_size]
        if bias_initializer is None:
            bias_initializer = tf.zeros_initializer
        else:
            if not callable(bias_initializer):
                bias_initializer = tf.convert_to_tensor(bias_initializer, tf.float32)
                bias_shape = None
        biases = tf.get_variable(
            name="biases",
            shape=bias_shape,
            initializer=bias_initializer,
            trainable=True
        )
        return tf.nn.bias_add(tf.matmul(inputs, weights), biases)


def max_pool(inputs, name=None):
    """Max pool with chip features"""
    return tf.nn.max_pool(
        value=inputs,
        name=name,
        ksize=[1, gti.chip.spec.POOLING_KERNEL, gti.chip.spec.POOLING_KERNEL, 1],
        strides=[1, gti.chip.spec.POOLING_STRIDE, gti.chip.spec.POOLING_STRIDE, 1],
        padding=gti.chip.spec.POOLING_PADDING
    )


def topleft_pool(inputs):
    """GTI device specific pooling method. Also known as sample pooling.

    Select top-left element in a 2x2 pooling kernel, with stride of 2. For example,
    Inputs:
        [*12. 12. *20.  4.]
        [ 9. 20. 22. 11.]
        [*4. 29. *2. 29.]
        [ 4. 17. 13. 21.]
    Outputs: the starred * elements are selected
        [12. 20.]
        [ 4.  2.]

    Args:
        inputs: 4-D tensor. Shape must be "NHWC", [batch, height, width, channels]. Height and width
        must be even numbers >= 14.
    Returns:
        4-D tensor. Output from top-left pooling.
    """
    return inputs[:, ::gti.chip.spec.POOLING_STRIDE, ::gti.chip.spec.POOLING_STRIDE, :]


def batch_norm(
        inputs,
        training,
        layer_scope,
        beta_initializer=None,
        gamma_initializer=None,
        moving_mean_initializer=None,
        moving_variance_initializer=None
    ):
    with tf.variable_scope(layer_scope):
        if beta_initializer is None:
            beta_initializer = tf.zeros_initializer
        else:
            if not callable(beta_initializer):
                beta_initializer = tf.constant_initializer(beta_initializer, verify_shape=True)
        if gamma_initializer is None:
            gamma_initializer = tf.ones_initializer
        else:
            if not callable(gamma_initializer):
                gamma_initializer = tf.constant_initializer(gamma_initializer, verify_shape=True)
        if moving_mean_initializer is None:
            moving_mean_initializer = tf.zeros_initializer
        else:
            if not callable(moving_mean_initializer):
                moving_mean_initializer = tf.constant_initializer(moving_mean_initializer, verify_shape=True)
        if moving_variance_initializer is None:
            moving_variance_initializer = tf.ones_initializer
        else:
            if not callable(moving_variance_initializer):
                moving_variance_initializer = tf.constant_initializer(moving_variance_initializer, verify_shape=True)
        return tf.layers.batch_normalization(
            inputs=inputs,
            training=training,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer
        )


BatchNormVars = namedtuple("BatchNormVars", ["mean", "variance", "gamma", "beta"])
def get_npy_bn_vars(npy, layer):
    """Get batch norm variables from numpy file.

    If batch norm is used in pretrained numpy file, variables are assumed to be saved according 
    to layer name. Each layer is assumed to have at index:
      0: weight
      1: moving mean
      2: moving variance
      3: gamma
      4: beta
    e.g. {"layer0": [weight, mean, variance, gamma, beta]}
    """
    if layer not in npy:
        _logger.warning("Layer: {} not found in numpy file".format(layer))
        return BatchNormVars(mean=None, variance=None, gamma=None, beta=None)
    if len(npy[layer]) != 5:
        raise Exception("Incompatible numpy file format for layer {}".format(layer))
    _, mean, variance, gamma, beta = npy[layer]
    return BatchNormVars(mean=mean, variance=variance, gamma=gamma, beta=beta)


def get_npy_weight(npy, layer):
    """Get weight variable from numpy file.

    Weight variable is expected to be saved according to layer name. weight is expected at index 0
    e.g. {"layer0": [weight, bias]}
    """
    if layer not in npy:
        _logger.warning("Layer: {} not found in numpy file".format(layer))
        return None
    weight = npy[layer][0]
    _logger.debug("Weight variable shape: {}".format(weight.shape))
    return weight


def get_npy_bias(npy, layer):
    """Get bias variable from numpy file.

    bias variable is expected to be saved according to layer name. bias is expected at index 1
    e.g. {"layer0": [weight, bias]}
    """
    if layer not in npy:
        _logger.warning("Layer: {} not found in numpy file".format(layer))
        return None
    if len(npy[layer]) != 2:
        raise Exception("Incompatible numpy file format for layer {}".format(layer))
    bias = npy[layer][1]
    _logger.debug("Bias variable shape: {}".format(bias.shape))
    return bias


def fuse_vars(checkpoint, previous_scope, current_scope, activation_bits=5):
    """Fuse batch norm and ReLU cap into weight and bias. Assume GTI variable scope convention."""
    w = checkpoint.get_tensor(current_scope + "/filters")

    # fuse batch norm
    try:  # fusing batch norm if exists
        bn = current_scope + "/batch_normalization/" 
        beta = checkpoint.get_tensor(bn + "beta")
        gamma = checkpoint.get_tensor(bn + "gamma")
        moving_mean = checkpoint.get_tensor(bn + "moving_mean")
        moving_variance = checkpoint.get_tensor(bn + "moving_variance")
        bn_epsilon = 1e-3  # default batch norm epislon
        bn_stddev = np.sqrt(moving_variance + bn_epsilon)
        bn_factor = gamma / bn_stddev
        # for depthwise convolution weights, reshape batch norm factor, so that
        # multiplication of the factor is not broadcasted to the last dimension
        if w.shape[3] == 1:
            bn_factor = bn_factor.reshape((1, 1, bn_factor.shape[0], 1))
        w *= bn_factor
        b = beta - gamma / bn_stddev * moving_mean
    except Exception:  # try get bias if already fused
        b = checkpoint.get_tensor(current_scope + "/biases")

    # fuse ReLU cap
    curr_cap = checkpoint.get_tensor(current_scope + "/relu_cap")[0]
    max_activation = gti.chip.spec.get_max_activation(num_bits=activation_bits)
    b_gain = max_activation / curr_cap
    if previous_scope is None:
        w_gain = max_activation / curr_cap 
    else:
        prev_cap = checkpoint.get_tensor(previous_scope + "/relu_cap")[0]
        w_gain = prev_cap / curr_cap
    w *= w_gain
    b *= b_gain
    return w, b


def _pad_like_chip(inputs, stride, filter_size):
    """HACK: pad inputs to mimick chip sample pooling for only specific cases.
    
    Might not work in general and fail in other cases.
    Args:
        inputs: tensor. input to be padded, must be in NHWC order.
        stride: integer. stride of convolution operation.
        filter_size: tuple of integer. (height, width) of convolution filter used.
    Returns:
        padded input and padding scheme
    """
    if filter_size == (1, 1):
        return inputs, "VALID"
    elif filter_size == (3, 3) and stride == 2:
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]])  # pad top & left
        return inputs, "VALID"
    else:
        return inputs, "SAME"