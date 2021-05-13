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
"""Convert full model, including host layers (e.g. pooling, fully-connected, labels)"""
import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from gti.chip import driver
from gti.converter import (
    cleanup,
    convert_chip_layers,
    convert_host_layers,
    concat_scopes,
    LayerToConvert,
    HostLayer,
    update_model_json
)
from gti.model.vgg16 import vgg16 


def main(args):
    # define file paths here:
    dat_json = os.path.join("nets", args.chip + "_vgg16_dat.json") 
    model_json = os.path.join("addon", args.chip + "_vgg16_fullmodel.json")
    out_model = os.path.join("addon", args.chip + "_vgg16_full.model")
    labels_file = os.path.join("data", "imagenet_labels.txt")  # replace with your own labels file

    # vgg16 specifics
    # ReLU cap variable in checkpoint for last chip layer
    last_relu_var = "vgg16/conv5_3/relu_cap"
    # reshape the first fully-connected layer weight to account for dimension order difference
    # between TensorFlow and chip
    reshape_to = (7, 7, 512, 4096)

    model = vgg16(target_chip=args.chip)
    layers_to_convert = []
    for layer_name in model.chip_layers:
        scope = concat_scopes(model.name, layer_name)
        weight = scope + "filters"
        mask_bit = model.mask_bit[layer_name]
        relu_cap = scope + "relu_cap"
        bias = scope + "biases"
        layers_to_convert.append(
            LayerToConvert(
                weight=weight,
                mask_bit=mask_bit,
                relu_cap=relu_cap,
                bias=bias
            )
        )
    save_dir = "addon"
    data_files = convert_chip_layers(
        checkpoint=args.checkpoint,
        layers_to_convert=layers_to_convert,
        target_chip=args.chip,
        dat_json=dat_json,
        save_dir=save_dir
    )
    fc_layers = []
    for layer_name in model.host_layers:
        scope = concat_scopes(model.name, layer_name)
        w_name = scope + "weights"
        b_name = scope + "biases"
        fc_layers.append(HostLayer(name=layer_name, weight=w_name, bias=b_name))
    data_files = convert_host_layers(
        checkpoint=args.checkpoint,
        host_layers=fc_layers,
        data_files=data_files,
        last_relu_var=last_relu_var,
        reshape_to=reshape_to,
        save_dir=save_dir
    )
    data_files["label"] = os.path.realpath(labels_file)
    update_model_json(model_json=model_json, data_files=data_files)
    if os.path.exists(out_model):
        print("{} already exists and will be overwritten".format(out_model))
    driver.compose_model(json_file=model_json, model_file=out_model)
    cleanup(save_dir=save_dir, data_files=data_files)
    print("successfully generated {}".format(out_model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint path"
    )
    parser.add_argument(
        "--chip",
        type=str,
        required=True,
        help="chip"
    )
    args = parser.parse_args()
    main(args)