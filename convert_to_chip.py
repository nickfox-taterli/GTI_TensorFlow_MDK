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

"""Example script to convert a quantized TensorFlow checkpoint to chip format."""
import os

from gti.converter import convert, concat_scopes, LayerToConvert
from gti.param_parser import ConversionParser
from gti.model.utils import model_factory, path_helper


def main(args):
    model = model_factory(name=args.net, chip=args.chip, checkpoint=args.checkpoint)
    # existence of batch norm is False for the following 2 conditions:
    # 1. model does not use batch norm, e.g. GTI vgg16 
    # 2. checkpoint was previously fused, thus no longer contains batch norm parameters
    if args.net in {"vgg16"} or args.fuse:
        batch_norm = False
    else:
        batch_norm = True

    # NOTE: if custom model checkpoint does not follow GTI variable scoping convention:
    #   [model name]/[layer name]/[variable name],
    # then instantiate LayerToConvert with custom variable names, e.g:
    #   LayerToConvert(weight="MyWeight", bias="MyBias", relu_cap="MyRelu"...)
    # Code below assumes model checkpoint follows GTI convention.
    layers_to_convert = []
    for layer_name in model.chip_layers:
        scope = concat_scopes(model.name, layer_name)
        weight = scope + "filters"
        mask_bit = model.mask_bit[layer_name]
        relu_cap = scope + "relu_cap"
        if batch_norm:
            bn_scope = concat_scopes(scope, "batch_normalization")
            bn_beta = bn_scope + "beta"
            bn_gamma = bn_scope + "gamma"
            bn_mean = bn_scope + "moving_mean"
            bn_variance = bn_scope + "moving_variance"
            layers_to_convert.append(
                LayerToConvert(
                    weight=weight,
                    mask_bit=mask_bit,
                    relu_cap=relu_cap,
                    bn_beta=bn_beta,
                    bn_gamma=bn_gamma,
                    bn_mean=bn_mean,
                    bn_variance=bn_variance
                )
            )
        else:
            bias = scope + "biases"
            layers_to_convert.append(
                LayerToConvert(
                    weight=weight,
                    mask_bit=mask_bit,
                    relu_cap=relu_cap,
                    bias=bias
                )
            )
    paths = path_helper(net=args.net, chip=args.chip)
    if args.ten_bit_act:
        act_bits = 10
    else:
        act_bits = 5
    convert(
        checkpoint=args.checkpoint,
        layers_to_convert=layers_to_convert,
        target_chip=args.chip,
        dat_json=paths.dat_json,
        model_json=paths.model_json,
        out_model=paths.model,
        activation_bits=act_bits
    )

if __name__ == "__main__":
    parser = ConversionParser() 
    args = parser.parse_args()
    main(args)
    