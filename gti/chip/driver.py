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
from ctypes import (
    byref, cast, CDLL, POINTER, Structure,
    c_float, c_ulonglong, c_char_p, c_int, c_void_p
)
import json
import os
import platform

import numpy as np

import gti.chip.spec
from gti.quantize import get_permute_axes


if platform.system() != "Linux":
    raise NotImplementedError("Windows support is currently limited") 
libgtisdk = CDLL(os.path.join(os.path.dirname(__file__), "libGTILibrary.so"))


class GtiTensor(Structure):
    _fields_ = [
        ("width", c_int),
        ("height", c_int),
        ("depth", c_int),
        ("stride", c_int),
        ("buffer", c_void_p),
        ("size", c_int),  # buffer size
        ("format", c_int)  # tensor format
    ]


class GtiModel(object):
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError("{} model file does not exist".format(model_path))
        libgtisdk.GtiCreateModel.argtypes = [c_char_p]
        libgtisdk.GtiCreateModel.restype = c_ulonglong
        self.obj = libgtisdk.GtiCreateModel(model_path.encode('ascii'))
    	
    def evaluate(self, numpy_array, last_relu_cap, activation_bits=5):
        """Evaluate tensor on GTI device for chip layers only.

        Args:
            numpy_array: 3D or 4D array in [(batch,) height, width, channel] order. Batch must be 1.
            last_relu_cap: float. the cap of the last activation layer on GTI device
        Returns:
            4D output numpy array in [batch, height, width, channel] order
        """
        if len(numpy_array.shape) == 4:  # squeeze batch dimension
            numpy_array = numpy_array.squeeze(axis=0)
        if len(numpy_array.shape) != 3:
            raise ValueError("Input dimension must be HWC or NHWC")

        # transform chip input tensor
        # 1. split tensor by depth/channels, e.g. BGR channels = 3
        # 2. vertically stack channels
        in_height, in_width, in_channels = numpy_array.shape
        numpy_array = np.vstack(np.dsplit(numpy_array, in_channels))
        in_tensor = GtiTensor(
            in_width,
            in_height,
            in_channels,
            0,  # stride = 0, irrelevant for this use case
            numpy_array.ctypes.data,  # input buffer
            in_channels * in_height * in_width,  # input buffer size
            0,  # tensor format = 0, binary format
        )

        libgtisdk.GtiEvaluate.argtypes = [c_ulonglong, POINTER(GtiTensor)]
        libgtisdk.GtiEvaluate.restype = POINTER(GtiTensor)
        out_tensor = libgtisdk.GtiEvaluate(self.obj, byref(in_tensor))

        # transform chip output tensor
        out_width = out_tensor.contents.width
        out_height = out_tensor.contents.height
        out_channels = out_tensor.contents.depth
        # output tensor is [channel, height, width] order
        out_shape = (1, out_channels, out_height, out_width)  # add 1 as batch dimension
        # for this use case, output tensor is floating point 
        out_buffer = cast(out_tensor.contents.buffer, POINTER(c_float))
        return (
            np.ctypeslib.as_array(out_buffer, shape=(np.prod(out_shape),))
            .reshape(out_shape)  # reshape buffer to 4D tensor
            .transpose(get_permute_axes("NCHW", "NHWC"))  # transpose to TensorFlow order
            / (gti.chip.spec.get_max_activation(num_bits=activation_bits) / last_relu_cap)  # divide chip gain
        )

    def full_inference(self, numpy_array):
        """Performs full inference, including chip and host layers defined in model JSON.

        Args:
            numpy_array: 3D or 4D array in [(batch,) height, width, channel] order. Batch must be 1.
        Returns:
            result: str. encoded as UTF-8 JSON.
        """
        if len(numpy_array.shape) == 4:  # squeeze batch dimension
            numpy_array = numpy_array.squeeze(axis=0)
        if len(numpy_array.shape) != 3:
            raise ValueError("Input dimension must be HWC or NHWC")
        height, width, channels = numpy_array.shape
        numpy_array = np.vstack(np.dsplit(numpy_array, channels))
        libgtisdk.GtiImageEvaluate.argtypes = (c_ulonglong, c_char_p, c_int, c_int, c_int)
        libgtisdk.GtiImageEvaluate.restype = c_char_p
        result = libgtisdk.GtiImageEvaluate(self.obj, numpy_array.tobytes(), height, width, channels)
        return json.loads(result.decode("utf-8"))
        
    def release(self):
        if self.obj is not None:
            libgtisdk.GtiDestroyModel.argtypes = [c_ulonglong]
            libgtisdk.GtiDestroyModel.restype = c_int
            destroyed = libgtisdk.GtiDestroyModel(self.obj)
            if not destroyed:
                raise Exception("Unable to release sources for GTI driver model")
            self.obj = None


def compose_model(json_file, model_file):
    libgtisdk.GtiComposeModelFile.argtypes = [c_char_p, c_char_p]
    return libgtisdk.GtiComposeModelFile(json_file.encode('ascii'), model_file.encode('ascii'))