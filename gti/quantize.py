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

"""Quantization functions."""
import tensorflow as tf

import gti.chip.spec

_EPSILON = 1e-6

def get_permute_axes(from_, to_):
    """ Get permutation axes from data ordering formats.

    For example:
        get_permute_axes('HWIO', 'OIHW') = [3, 2, 0, 1]
    will set the transpose axes from TensorFlow ordering to GTI ordering, i.e.
    from:
        [filter height, filter width, input channels, output channels]
    to:
        [output channels, input channels, filter height, filter width]

    Args: 
        from_: From data order format string.
        to_: To data order format string.

    Returns:
        permutation axes list.
    """
    return list(map(lambda x: from_.find(x), to_))


def quantize_filters(filters, num_bits, shift):
    if num_bits == 1:
        return _quantize_1bit(filters, shift)
    elif num_bits == 2:
        return _quantize_2bit(filters, shift)
    elif num_bits == 3:
        return _quantize_3bit(filters, shift)
    elif num_bits == 5:
        return _quantize_5bit(filters, shift)
    elif num_bits in {8, 12}:
        return _quantize_morebit(filters, num_bits, shift)
    else:
        raise ValueError("Currently {}-bit quantization is not supported".format(num_bits))



def quantized_conv2d(inputs, filters, biases, target_chip, mask_bitwidth, strides, padding):
    shift = compute_shift(
        filters=filters,
        biases=biases,
        target_chip=target_chip,
        mask_bitwidth=mask_bitwidth
    )
    filters = quantize_filters(filters, mask_bitwidth, shift)
    if biases is None:  # not using bias terms
        return tf.nn.conv2d(inputs, filters, strides, padding) 
    biases = quantize_shift(biases, shift) 
    return tf.nn.bias_add(tf.nn.conv2d(inputs, filters, strides, padding), biases)


def quantized_depthwise_conv2d(inputs, filters, biases, target_chip, mask_bitwidth, strides, padding):
    shift = compute_shift(
        filters=filters,
        biases=biases,
        target_chip=target_chip,
        mask_bitwidth=mask_bitwidth
    )
    filters = quantize_filters(filters, mask_bitwidth, shift)
    if biases is None:  # not using bias terms
        return tf.nn.depthwise_conv2d(inputs, filters, strides, padding) 
    biases = quantize_shift(biases, shift)
    return tf.nn.bias_add(tf.nn.depthwise_conv2d(inputs, filters, strides, padding), biases)


def quantized_relu(inputs, cap, activation_bits=5, trainable=False):
    max_activation = gti.chip.spec.get_max_activation(num_bits=activation_bits)
    regularizer = tf.keras.regularizers.l2(l=0.01) if trainable else None
    relu_cap = tf.get_variable(
        name="relu_cap",
        initializer=tf.constant([cap], dtype=tf.float32),
        regularizer=regularizer,
        trainable=trainable
    )
    y = 0.5 * (tf.abs(inputs) - tf.abs(inputs - relu_cap) + relu_cap)
    y = _round(y * max_activation / relu_cap) * relu_cap / max_activation 
    return y


def quantize_shift(x, shift):
    """Quantize tensor by shift mechanism."""
    y = _round(x * (2 ** shift)) / (2 ** shift)
    return x + tf.stop_gradient(y - x)


def compute_shift(filters, biases, target_chip, mask_bitwidth):
    weight_bitwidth, bias_bitwidth = gti.chip.spec.get_quantization_scheme(target_chip, mask_bitwidth)
    weight_shift = _bit_shift_helper(filters, weight_bitwidth)
    if biases is None:
        shift = tf.clip_by_value(weight_shift, gti.chip.spec.MIN_SHIFT, gti.chip.spec.MAX_SHIFT)
        return tf.stop_gradient(shift)
    bias_shift = _bit_shift_helper(biases, bias_bitwidth)
    shift = tf.clip_by_value(
        tf.minimum(weight_shift, bias_shift),
        gti.chip.spec.MIN_SHIFT, gti.chip.spec.MAX_SHIFT
    )
    return tf.stop_gradient(shift)


def _bit_shift_helper(x, max_bitwidth):
    steps = (2.0 ** (max_bitwidth - 1) - 1) / tf.reduce_max(tf.abs(x))
    shift = tf.cast(tf.log(steps) / tf.log(2.0), tf.int32)
    shift = tf.cast(shift, tf.float32)
    return tf.stop_gradient(shift)


def _quantize_1bit(x, shift):
    """Quantize to {-1, 1}."""
    xt = tf.transpose(x, perm=get_permute_axes("HWIO", "OIHW"))
    mean_abs = tf.reduce_mean(tf.abs(xt), axis=[2, 3], keepdims=True)
    mean_abs = quantize_shift(mean_abs, shift)
    y = tf.where(xt >= 0, tf.ones(tf.shape(xt)) * mean_abs, tf.ones(tf.shape(xt)) * -mean_abs)
    y = tf.transpose(y, perm=get_permute_axes("OIHW", "HWIO"))
    return x + tf.stop_gradient(y - x)


def _quantize_2bit(x, shift):
    """Quantize to {-1, 0, 1}."""
    xt = tf.transpose(x, perm=get_permute_axes("HWIO", "OIHW"))
    abs_xt = tf.abs(xt)
    mean_abs = tf.reduce_mean(abs_xt, axis=[2, 3], keepdims=True)
    mean_abs = quantize_shift(mean_abs, shift)
    y = tf.where(abs_xt >= mean_abs / 4, tf.ones(tf.shape(xt)) * mean_abs, tf.zeros(tf.shape(xt)))
    y = tf.where(xt >= 0, y, -y)
    y = tf.transpose(y, perm=get_permute_axes("OIHW", "HWIO"))
    return x + tf.stop_gradient(y - x)


def _quantize_3bit(x, shift):
    """Quantize to {-4, -2, -1, 0, 1, 2, 4}."""
    xt = tf.transpose(x, perm=get_permute_axes("HWIO", "OIHW"))
    abs_xt = tf.abs(xt)
    mean_abs = tf.reduce_mean(abs_xt, [2, 3], keepdims=True)
    step = tf.where(mean_abs == 0, tf.ones(tf.shape(mean_abs)) * _EPSILON, mean_abs) / 4.0
    coef = tf.cast(tf.cast(abs_xt / step, tf.int32), tf.float32)
    step = quantize_shift(step, shift)
    y = tf.where(coef >= 3, tf.fill(tf.shape(coef), 4.0), coef)
    y = tf.where(xt >= 0, y * step, -y * step)
    y = tf.transpose(y, perm=get_permute_axes("OIHW", "HWIO"))
    return x + tf.stop_gradient(y - x)


def _quantize_5bit(x, shift):
    """Quantize to [-15, 15]."""
    xt = tf.transpose(x, perm=get_permute_axes("HWIO", "OIHW"))
    max_abs = tf.reduce_max(tf.abs(xt), [2, 3], keepdims=True)
    step = tf.where(max_abs == 0, tf.ones(tf.shape(max_abs)) * _EPSILON, max_abs) / 15.0
    coef = tf.cast(tf.cast(xt / step, tf.int32), tf.float32)
    step = quantize_shift(step, shift)
    y = coef * step
    y = tf.transpose(y, perm=get_permute_axes("OIHW", "HWIO"))
    return x + tf.stop_gradient(y - x)


def _quantize_morebit(x, num_bits, shift):
    """Quantize by N bit symmetrically and uniformly, where 6 <= N <= 31"""
    if num_bits < 6 or num_bits > 31:
        raise ValueError("Invalid quantization scheme for {} bits".format(num_bits))
    xt = tf.transpose(x, perm=get_permute_axes("HWIO", "OIHW"))
    y = quantize_shift(xt, shift)
    y = tf.transpose(y, perm=get_permute_axes("OIHW", "HWIO"))
    return x + tf.stop_gradient(y - x)


def _round(x):
    """Simulate chip rounding: half-up 2.5 -> 3, not using default half-to-even 2.5 -> 2."""
    y = tf.where(x >= 0, tf.floor(x + 0.5), tf.ceil(x - 0.5))
    return x + tf.stop_gradient(y - x)