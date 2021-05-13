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

"""
GTI VGG16
"""
from collections import namedtuple

import tensorflow as tf

import gti.chip.spec
import gti.layers
from gti.layers import get_npy_weight, get_npy_bias
from gti.model.base import BaseModel


VGGBlock = namedtuple("VGGBlock", ["name", "size", "in_channels", "out_channels"])
VGG16_CONV_BLOCKS_DEF = [
    VGGBlock(name="conv1", size=2, in_channels=3, out_channels=64),
    VGGBlock(name="conv2", size=2, in_channels=64, out_channels=128),
    VGGBlock(name="conv3", size=3, in_channels=128, out_channels=256),
    VGGBlock(name="conv4", size=3, in_channels=256, out_channels=512),
    VGGBlock(name="conv5", size=3, in_channels=512, out_channels=512)
] 

class vgg16(BaseModel):
    """GTI VGG16."""
    def __init__(
            self,
            target_chip,
            classes=None,
            quant_w=False,
            quant_act=False,
            fuse=False,
            checkpoint=None,
            npy=None
        ):
        # Naming convention: conv[MAJOR LAYER ID]_[SUBLAYER ID]. IDs start at 1.
        chip_layers = []
        for block in VGG16_CONV_BLOCKS_DEF:
            for sub_id in range(block.size):
                chip_layers.append(block.name + "_" + str(sub_id + 1))
        super(vgg16, self).__init__(
            name="vgg16",
            chip_layers=chip_layers,
            host_layers=["fc6", "fc7", "fc8"],
            target_chip=target_chip,
            chip_output_shape=(7, 7, 512),  # (height, width, channel)
            quant_w=quant_w,
            quant_act=quant_act,
            fuse=fuse,
            classes=classes,
            checkpoint=checkpoint,
            npy=npy
        )

    def _vgg_block(
            self,
            inputs,
            block_size,
            in_channels,
            out_channels,
            layer_scope,
        ):
        """VGG block with [block_size] convolutional layers, each followed by a ReLU, and
        finally by max pool."""
        for sublayer_id in range(block_size):
            sublayer_scope = layer_scope + "_" + str(sublayer_id + 1)
            if sublayer_id == 0:
                in_channels = in_channels
            else: # input channels == output channels for 2nd sublayer onwards
                in_channels = out_channels
            
            if self.fuse:
                scope = self.name + "/" + sublayer_scope
                w = self.checkpoint.get_tensor(scope + "/filters")
                b = self.checkpoint.get_tensor(scope + "/biases")
                curr_cap = self.checkpoint.get_tensor(scope + "/relu_cap")[0]
                b_gain = gti.chip.spec.get_max_activation() / curr_cap
                layer_idx = self.chip_layers.index(sublayer_scope)
                if layer_idx == 0:  # first layer
                    w_gain = b_gain 
                else:
                    prev_scope = self.chip_layers[layer_idx - 1]
                    prev_cap = self.checkpoint.get_tensor(self.name + "/" + prev_scope + "/relu_cap")[0]
                    w_gain = prev_cap / curr_cap
                w *= w_gain
                b *= b_gain
            elif self.npy is not None:
                w = get_npy_weight(npy=self.npy, layer=sublayer_scope)
                b = get_npy_bias(npy=self.npy, layer=sublayer_scope)
            else:
                w = b = None
            inputs = gti.layers.conv2d(
                inputs=inputs,
                in_channels=in_channels,
                out_channels=out_channels,
                layer_scope=sublayer_scope,
                quantize=self.quant_w,
                target_chip=self.target_chip,
                mask_bitwidth=self.mask_bit[sublayer_scope],
                filter_initializer=w,
                bias_initializer=b
            )
            if self.fuse:
                # if fuse, gti.layers.relu trainable must be set False
                cap = gti.chip.spec.get_max_activation()
            else:
                cap = self.relu_cap[sublayer_scope]
            inputs = gti.layers.relu(
                inputs=inputs,
                layer_scope=sublayer_scope,
                quantize=self.quant_act,
                cap=cap
            )
            self.reference_relu_output(layer_scope=sublayer_scope, value=inputs)
        return gti.layers.max_pool(inputs)


    def build_chip_net(self, inputs, is_training):
        """Build portion of network that will be on-chip.
        In general, the convolutional layers will be on-chip. 
        """
        for block in VGG16_CONV_BLOCKS_DEF:
            inputs = self._vgg_block(
                inputs=inputs,
                block_size=block.size,
                in_channels=block.in_channels,
                out_channels=block.out_channels,
                layer_scope=block.name,
            )
        return inputs
    

    def build_host_net(self, inputs, is_training):
        """Build portion of network that will be on-host.
        In general, global average pooling & fully-connected layers will be on-host.
        """
        # Flatten convolutional network output for fully-connected layers next
        self.flatten = tf.layers.flatten(inputs)
        if self.fuse:  # fuse batch norm and ReLU cap, init from checkpoint
            last_relu_var = self.name + "/" + self.chip_layers[-1] + "/relu_cap"
            scale_w = gti.chip.spec.get_max_activation() / self.checkpoint.get_tensor(last_relu_var)[0]
            w6 = self.checkpoint.get_tensor(self.name + "/" + self.host_layers[0] + "/weights") / scale_w 
            b6 = self.checkpoint.get_tensor(self.name + "/" + self.host_layers[0] + "/biases")
            w7 = self.checkpoint.get_tensor(self.name + "/" + self.host_layers[1] + "/weights")
            b7 = self.checkpoint.get_tensor(self.name + "/" + self.host_layers[1] + "/biases")
            w8 = self.checkpoint.get_tensor(self.name + "/" + self.host_layers[2] + "/weights")
            b8 = self.checkpoint.get_tensor(self.name + "/" + self.host_layers[2] + "/biases")
        elif self.npy is not None:  # init from numpy file
            w6 = get_npy_weight(npy=self.npy, layer=self.host_layers[0])
            b6 = get_npy_bias(npy=self.npy, layer=self.host_layers[0])
            w7 = get_npy_weight(npy=self.npy, layer=self.host_layers[1]) 
            b7 = get_npy_bias(npy=self.npy, layer=self.host_layers[1])
            w8 = get_npy_weight(npy=self.npy, layer=self.host_layers[2]) 
            b8 = get_npy_bias(npy=self.npy, layer=self.host_layers[2])
        else:  # default init 
            w6 = b6 = w7 = b7 = w8 = b8 = None

        # FC Block 6
        self.fc6 = gti.layers.fully_connected(
            inputs=self.flatten,
            in_size=7*7*512,
            out_size=4096,
            layer_scope=self.host_layers[0],
            weight_initializer=w6,
            bias_initializer=b6
        )
        self.relu6 = tf.nn.relu(self.fc6)
        self.relu6 = tf.layers.dropout(self.relu6, training=is_training)

        # FC Block 7
        self.fc7 = gti.layers.fully_connected(
            inputs=self.relu6,
            in_size=4096,
            out_size=4096,
            layer_scope=self.host_layers[1],
            weight_initializer=w7,
            bias_initializer=b7
        )
        self.relu7 = tf.nn.relu(self.fc7)
        self.relu7 = tf.layers.dropout(self.relu7, training=is_training)

        # FC Block 8
        if w8 is not None or b8 is not None:  # if conflict in pretrained, use user-defined classes
            if self.classes != w8.shape[-1] or self.classes != b8.shape[-1]:
                w8 = b8 = None
        self.fc8 = gti.layers.fully_connected(
            inputs=self.relu7,
            in_size=4096,
            out_size=self.classes,
            layer_scope=self.host_layers[2],
            weight_initializer=w8,
            bias_initializer=b8
        )
        return self.fc8


    def build(self, inputs, is_training):
        out = self.build_chip_net(inputs=inputs, is_training=is_training)
        return self.build_host_net(inputs=out, is_training=is_training)