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
import json
import os

import numpy as np
import tensorflow as tf


class BaseModel(object):
    def __init__(
            self,
            name,
            chip_layers,
            host_layers,
            target_chip,
            chip_output_shape,
            quant_w=False,
            quant_act=False,
            fuse=False,
            classes=None,
            checkpoint=None,
            npy=None
        ):
        target_chip = str(target_chip)  # sanitize user input in case integers
        if checkpoint is not None and npy is not None:
            raise ValueError("Ambiguous: use checkpoint or numpy file, not both")
        if fuse and checkpoint is None:
            raise ValueError("Must specify checkpoint if fusing batch norm and ReLu caps")
        if len(chip_output_shape) == 3:  # add batch dimension
            self.chip_output_shape = (1,) + chip_output_shape
        elif len(chip_output_shape) == 4:
            self.chip_output_shape = chip_output_shape
        else:
            raise ValueError("Invalid chip output shape, must be tuple of 3 or 4 int")
        settings_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "settings",
            target_chip + "_" + name + ".json"
        ) 
        with open(settings_path, "r") as f:
            settings = json.load(f)
            self.relu_cap = settings["relu_cap"]
            self.mask_bit = settings["mask_bit"]
        if checkpoint is None:
            self.checkpoint = None
        else:
            self.checkpoint = tf.train.NewCheckpointReader(checkpoint)
        if npy is None:
            self.npy = None
        else:
            self.npy = np.load(npy, encoding="latin1").item()
        self.name = name
        self.chip_layers = chip_layers
        self.relu_outputs = {k: None for k in self.chip_layers}
        self.host_layers = host_layers
        self.target_chip = target_chip
        self.quant_w = quant_w
        self.quant_act = quant_act
        self.fuse = fuse
        self.classes = classes
        self.settings_path = os.path.realpath(settings_path)

    def reference_relu_output(self, layer_scope, value):
        """Reference ReLU output node in graph for subsequent quantization steps."""
        if layer_scope not in self.relu_outputs:
            raise ValueError("Layer scope: {} not defined in ReLU outputs".format(layer_scope))
        self.relu_outputs[layer_scope] = value 