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
from enum import Enum
import os
import json

# Supported input image sizes, height == width
DEFAULT_IMAGE_SIZE = 224
ALLOWED_IMAGE_SIZES = {224, 448}

DEFAULT_CONVOLUTION_FILTER_SIZE = (3, 3)
ALLOWED_CONVOLUTION_FILTER_SIZES = {(1, 1), (3, 3)}
ALLOWED_CONVOLUTION_STRIDES = {1, 2}

# Pooling kernel and stride
POOLING_KERNEL = POOLING_STRIDE = 2
POOLING_PADDING = "VALID"

# Shift
MIN_SHIFT = 0
MAX_SHIFT = 12


MAX_ACTIVATION_VALUE = {
    # number of bits: max representable value
    5: 31.0,
    10: 1023.0
}


TEN_BIT_RELU_CAP = 31.96875

class UpSamplingFillMode(Enum):
    REPEAT = 1
    ZERO = 2

def get_max_activation(num_bits=5):
    """Get max activation value.

    Args:
        num_bits (int): number of bits to use for activation
    Returns:
        maximum representable value (float)

    5-bit: default, representable range [0, 31]
    10-bit: option, representable range [0, 1023], but must reduce number of channels
    """
    try:
        return MAX_ACTIVATION_VALUE[num_bits] 
    except KeyError:
        raise ValueError("Invalid number of activation bits")


def get_quantization_scheme(chip, mask_bit):
    """Get quantization scheme based on chip spec.

    Args:
        chip (str): chip ID
        mask_bit (int): number of bits for quantizing convolution filters
    Returns:
        weight bit (int), bias bit(int)
    """
    specs = _get_specs(chip)
    mask_bit = str(mask_bit)
    weight_bit = specs["quantization"][mask_bit]["weight"]
    bias_bit = specs["quantization"][mask_bit]["bias"]
    return weight_bit, bias_bit


def _get_specs(chip):
    chip = str(chip)
    spec_path = os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            chip + "_spec.json"
        )
    )
    with open(spec_path, "r") as f:
        specs = json.load(f)
        return specs
