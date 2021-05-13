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

"""Convert TensorFlow checkpoint to chip format."""
import json
import logging
import os
import shutil

import numpy as np
import tensorflow as tf

import gti.chip.spec
from gti.chip import driver
from gti.config import gticonfig
from gti.quantize import quantize_filters, get_permute_axes, compute_shift, quantize_shift


# When on-chip model performance is not satisfactory, you may set _DEBUG_CONVERSION to True to see
# more details during conversion by setting environment variable before running conversion script:
#   GTI_DEBUG_CONVERSION=True python convert_to_chip.py
_DEBUG_CONVERSION = os.environ.get("GTI_DEBUG_CONVERSION") == "True" 
# if debug mode is on, log more details 
_CONVERISON_LOG_LEVEL = logging.DEBUG if _DEBUG_CONVERSION else logging.INFO
logging.basicConfig()
_logger = logging.getLogger(__name__)
_logger.setLevel(_CONVERISON_LOG_LEVEL)


class LayerToConvert(object):
    """Definition of a layer to be converted and accelerated on-chip 

    Bundles related checkpoint variable names together into an on-chip "layer" definition. A "layer"
    usually consists of:
        convolution-->batch norm-->ReLU
    To get a list of variable names from a checkpoint, use:
        tf.train.NewCheckpointReader(checkpoint).get_variable_to_shape_map()
    which returns a dictionary whose keys are the variable names.
    
    Attributes:
        weight (str): name of weight variable in checkpoint, e.g. "vgg16/conv1_1/filters"
        mask_bit (int): quantization mask bit used for weight variable, e.g. 3
        relu_cap (str): name of ReLU cap variable in checkpoint, e.g. "vgg16/conv1_1/relu_cap"
        bias (str): name of bias variable in checkpoint, e.g. "vgg16/conv1_1/biases" 
        bn_beta (str): name of batch norm beta variable in checkpoint, e.g. "vgg16/conv1_1/batch_normalization/beta"
        bn_gamma (str): name of batch norm gamma variable in checkpoint, e.g. "vgg16/conv1_1/batch_normalization/gamma"
        bn_mean (str): name of batch norm moving mean variable in checkpoint, e.g. "vgg16/conv1_1/batch_normalization/moving_mean"
        bn_variance (str): name of batch norm beta variable in checkpoint, e.g. "vgg16/conv1_1/batch_normalization/moving_variance"
        bn_epsilon (float): batch norm epsilon value used in TF batch norm layer, default: 1e-3. If
            used different value during training, then set the same value here.
    """
    def __init__(
        self,
        weight,
        mask_bit,
        relu_cap,
        bias=None,
        bn_beta=None,
        bn_gamma=None,
        bn_mean=None, 
        bn_variance=None,
        bn_epsilon=1e-3
    ):
        batch_norm_used = (
            bn_beta is not None
            and bn_gamma is not None
            and bn_mean is not None
            and bn_variance is not None
            and bn_epsilon is not None
        )
        if not batch_norm_used and bias is None:
            raise ValueError("Must provide either batch norm variable names or bias variable name")
        if batch_norm_used and bias is not None:
            raise ValueError("If using batch norm, then bias should not be used")
        self.batch_norm_used = batch_norm_used
        self.weight = weight
        self.mask_bit = mask_bit
        self.relu_cap = relu_cap
        self.bias = bias
        self.bn_beta = bn_beta
        self.bn_gamma = bn_gamma
        self.bn_mean = bn_mean
        self.bn_variance = bn_variance
        self.bn_epsilon = bn_epsilon


class HostLayer(object):
    """Host layer definition.
    
    Attributes:
        name (str): layer name, e.g. fc
        weight (str): weight variable name, e.g. vgg16/fc6/weights
        bias (str): bias variable name, e.g. vgg16/fc6/biases
    """
    def __init__(self, name, weight, bias):
        self.name = name
        self.weight = weight
        self.bias = bias


def convert(
        checkpoint,
        layers_to_convert,
        target_chip,
        dat_json,
        model_json,
        out_model,
        activation_bits=5
    ):
    """Convert checkpoint to chip-compatible .model

    Args:
        checkpoint (str): path of checkpoint, e.g. vgg16-qw-qa-epoch-100
        layers_to_convert (list of obj): list of LayerToConvert objects to be converted as
            chip-compatible model. Must be in same sequential order as defined in original model.
        target_chip (str): GTI chip for model to run on. Model must have been trained
            and quantized for the same chip.
        dat_json (str): path of DAT definition JSON
        model_json (str): path of MODEL definition JSON
        out_model (str): path of output model to be generated
        activation_bits (int): number of bits for activation on-chip for last layer
    
    Returns:
        None. Generate output model and write to disk.
    """
    save_dir = "nets"  # default folder to save intermediate files
    data_files = convert_chip_layers(
        checkpoint=checkpoint,
        layers_to_convert=layers_to_convert,
        target_chip=target_chip,
        dat_json=dat_json,
        save_dir=save_dir,
        activation_bits=activation_bits
    )
    # generate .model file 
    update_model_json(model_json=model_json, data_files=data_files)
    if os.path.exists(out_model):
        _logger.warning("{} already exists and will be overwritten".format(out_model))
    driver.compose_model(json_file=model_json, model_file=out_model)
    cleanup(save_dir, data_files)
    _logger.info("successfully generated {}".format(out_model))


def cleanup(save_dir, data_files):
    if not _DEBUG_CONVERSION:
        _logger.info("removing intermediate files generated during conversion")
        for k, v in data_files.items(): 
            if os.path.exists(v) and "label" not in k:
                os.remove(v)
    tb_out = os.path.join(save_dir, "gti.tb")
    if os.path.exists(tb_out):
        os.remove(tb_out)
    filter_dump_dirs = [
        os.path.join(save_dir, "filter_cmpr"),
        os.path.join(save_dir, "filter_debug"),
        os.path.join(save_dir, "infile_cmpr")
    ]
    for d in filter_dump_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)


def convert_chip_layers(
        checkpoint,
        layers_to_convert,
        target_chip,
        dat_json,
        save_dir,
        activation_bits=5
    ):
    """Convert chip layers into .DAT file

    Args:
        checkpoint (str): path of checkpoint, e.g. vgg16-qw-qa-epoch-100
        layers_to_convert (list of obj): list of LayerToConvert objects to be converted as
            chip-compatible model. Must be in same sequential order as defined in original model.
        target_chip (str): GTI chip for model to run on. Model must have been trained and quantized
            for the same chip.
        dat_json (str): path of DAT definition JSON
        save_dir (str): directory to save intermediate files
        activation_bits (int): number of bits for activation on-chip for last layer
    
    Returns:
        dictionary containing paths to data files, look up by key 
    """
    target_chip = str(target_chip)  # sanitize user input
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filter_file = os.path.join(save_dir, "filter.txt")
    bias_file = os.path.join(save_dir, "bias.txt")
    dat_out = os.path.join(save_dir, "chip.dat")

    # generate chip DAT
    flat_filter = np.array([])
    flat_bias = np.array([])
    bit_shifts = []
    with tf.Session() as sess:
        reader = tf.train.NewCheckpointReader(checkpoint)
        for idx, layer in enumerate(layers_to_convert):
            weight = reader.get_tensor(layer.weight)
            # calculate batch norm parameters & bias
            if not layer.batch_norm_used:
                bn_factor = 1
                bias = reader.get_tensor(layer.bias)
            else:
                bn_beta = reader.get_tensor(layer.bn_beta)
                bn_gamma = reader.get_tensor(layer.bn_gamma)
                bn_moving_mean = reader.get_tensor(layer.bn_mean)
                bn_moving_variance = reader.get_tensor(layer.bn_variance)
                bn_epsilon = layer.bn_epsilon
                bn_stddev = np.sqrt(bn_moving_variance + bn_epsilon)
                bn_factor = bn_gamma / bn_stddev
                # for depthwise convolution weights, reshape batch norm factor, so that
                # multiplication of the factor is not broadcasted to the last dimension
                if weight.shape[3] == 1:
                    bn_factor = bn_factor.reshape((1, 1, bn_factor.shape[0], 1))
                bias = bn_beta - bn_gamma / bn_stddev * bn_moving_mean
            
            # calculate gain
            current_relu_cap = reader.get_tensor(layer.relu_cap)[0]
            bias_gain = gti.chip.spec.get_max_activation() / current_relu_cap
            if idx == 0:  # 1st layer, same gain
                weight_gain = bias_gain 
            else:
                prev_relu_cap = reader.get_tensor(layers_to_convert[idx - 1].relu_cap)[0]
                if activation_bits == 10 and idx == len(layers_to_convert) - 1: # last layer is 10 bit 
                    if abs(current_relu_cap - gti.chip.spec.TEN_BIT_RELU_CAP) < 1e-4:
                        bias_gain = weight_gain = 1.0  # 10 bit layer already fused
                    else:
                        bias_gain = gti.chip.spec.TEN_BIT_RELU_CAP / current_relu_cap
                        weight_gain = (
                            (prev_relu_cap / gti.chip.spec.get_max_activation())
                            * (gti.chip.spec.TEN_BIT_RELU_CAP / current_relu_cap)
                        )
                else:
                    weight_gain = prev_relu_cap / current_relu_cap 
            # merge batch norm
            weight *= bn_factor

            # merge gain
            weight *= weight_gain
            bias *= bias_gain

            # calculate shift
            bit_shift = sess.run(
                compute_shift(
                    filters=weight,
                    biases=bias,
                    target_chip=target_chip,
                    mask_bitwidth=layer.mask_bit
                )
            )

            # quantize filters
            weight = sess.run(quantize_filters(filters=weight, num_bits=layer.mask_bit, shift=bit_shift))
            
            # shift quantize bias
            bias = sess.run(quantize_shift(bias, bit_shift))

            # NOTE: transpose filter weights from TensorFlow order:
            #       [filter_height, filter_width, in_channels, out_channels
            #   to GTI/PyTorch/Caffe order:
            #       [out_channels, in_channels, filter_height, filter_width]
            weight = weight.transpose(tuple(get_permute_axes("HWIO", "OIHW")))

            # Log detailed information for layer gains and parameter magnitudes
            _logger.debug("Layer: {}, {}".format(idx, layer.weight))
            _logger.debug("W gain: {}, B gain: {}".format(weight_gain, bias_gain))
            _logger.debug(
                "|W|max: {}, |B|max: {}, Shift: {}"
                .format(np.amax(np.absolute(weight)), np.amax(np.absolute(bias)), bit_shift)
            )
            _logger.debug("")
            flat_filter = np.concatenate((flat_filter, weight.ravel()))
            flat_bias = np.concatenate((flat_bias, bias.ravel()))
            bit_shifts.append(bit_shift)
    _logger.info("converting convolutional layers to .DAT file")
    flat_filter.tofile(filter_file, sep="\n", format="%.16e")
    flat_bias.tofile(bias_file, sep="\n", format="%.16e")
    update_dat_json(dat_json=dat_json, new_shifts=bit_shifts)

    with open(dat_json, "r") as f:
        src = json.load(f)
        src_model = src["model"][0]

        if "ChipNumber" in src_model and src_model["ChipNumber"] > 1:
            _logger.info("converting a multi-chip model")
            num_chips = src_model["ChipNumber"]
            filter_size = np.prod(gti.chip.spec.DEFAULT_CONVOLUTION_FILTER_SIZE)
            filtertxt = np.fromfile(filter_file, dtype=np.float32, sep="\n")
            biastxt = np.fromfile(bias_file, dtype=np.float32, sep="\n")

            if num_chips == 2:  # 2-chip
                # major layer split points:
                # chip0: [1 to 7, merge 6 and 7]
                # chip1: [8 to 10] 
                split_at_major_layer = 6  # hardcoded
                chip0_model = dict(src_model)
                chip0_model["ChipNumber"] = 1
                chip0_model["MajorLayerNumber"] = split_at_major_layer
                chip0_layers = [i for i in src["layer"][:split_at_major_layer - 1]]

                # merge chip0 last 2 layers
                chip0_last_layer = src["layer"][split_at_major_layer - 1]
                chip0_merge_layer = src["layer"][split_at_major_layer]
                chip0_last_layer["sublayer_number"] += chip0_merge_layer["sublayer_number"]
                chip0_last_layer["resnet_shortcut_start_layers"] = (
                    chip0_last_layer["resnet_shortcut_start_layers"]  # [1, 3, 5]
                    + [i + 6 for i in chip0_merge_layer["resnet_shortcut_start_layers"]] # [7, 9, 11]
                )
                chip0_last_layer["scaling"] += chip0_merge_layer["scaling"]
                chip0_layers.append(chip0_last_layer)

                chip1_model = dict(src_model)
                chip1_model["ChipNumber"] = 1
                chip1_model["MajorLayerNumber"] = 3
                chip1_layers = [i for i in src["layer"][split_at_major_layer + 1:]]
                
                jsons = [
                    {"model": [chip0_model], "layer": chip0_layers},
                    {"model": [chip1_model], "layer": chip1_layers}
                ]

            elif num_chips == 4:
                # major layer split points:
                # chip0: [1 to 5]
                # chip1: [6]
                # chip2: [7]
                # chip3: [8 to 10]
                chip0_model = dict(src_model)
                chip0_model["ChipNumber"] = 1
                chip0_model["MajorLayerNumber"] = 5 
                chip0_layers = [i for i in src["layer"][:5]]

                chip1_model = dict(src_model)
                chip1_model["ChipNumber"] = 1
                chip1_model["MajorLayerNumber"] = 1 
                chip1_layers = [src["layer"][5]]

                chip2_model = dict(src_model)
                chip2_model["ChipNumber"] = 1
                chip2_model["MajorLayerNumber"] = 1 
                chip2_layers = [src["layer"][6]]

                chip3_model = dict(src_model)
                chip3_model["ChipNumber"] = 1
                chip3_model["MajorLayerNumber"] = 3 
                chip3_layers = [i for i in src["layer"][7:]]

                jsons = [
                    {"model": [chip0_model], "layer": chip0_layers},
                    {"model": [chip1_model], "layer": chip1_layers},
                    {"model": [chip2_model], "layer": chip2_layers},
                    {"model": [chip3_model], "layer": chip3_layers},
                ]

            # initialize split starting and end points for filter and bias
            filter_start = filter_end = bias_start = bias_end = 0
            result = {}
            _logger.info("splitting JSON, filter file, and bias file")
            for idx, js in enumerate(jsons):
                with open(dat_json[:-5] + "_chip" + str(idx) + ".json", "w") as chipjson:
                    for lidx, l in enumerate(js["layer"]):  # update major layer index
                        l["major_layer"] = lidx + 1
                    json.dump(js, chipjson, indent=4, sort_keys=True)
                for l in js["layer"]:
                    filter_end += l["input_channels"] * l["output_channels"] * filter_size
                    for _ in l["scaling"][1:]:
                        filter_end += l["output_channels"] * l["output_channels"] * filter_size
                    bias_end += l["output_channels"] * len(l["scaling"])
                filtertxt[filter_start:filter_end].tofile(filter_file[:-4] + "_chip" + str(idx) + ".txt", sep="\n")
                biastxt[bias_start:bias_end].tofile(bias_file[:-4] + "_chip" + str(idx) + ".txt", sep="\n")
                filter_start = filter_end  # update filter split starting point
                bias_start = bias_end  # update bias split starting point
                json_i = dat_json[:-5] + "_chip" + str(idx) + ".json" 
                filter_i = filter_file[:-4] + "_chip" + str(idx) + ".txt"
                bias_i = bias_file[:-4] + "_chip" + str(idx) + ".txt"
                dat_i = dat_out[:-4] + str(idx) + ".dat"
                gticonfig(
                    dat_json=json_i,
                    filter_file=filter_i,
                    bias_file=bias_i,
                    dat_out=dat_i,
                    save_dir=save_dir
                )
                result["json" + str(idx)] = os.path.realpath(json_i)
                result["filter" + str(idx)] =  os.path.realpath(filter_i)
                result["bias" + str(idx)] = os.path.realpath(bias_i)
                result["dat" + str(idx)] = os.path.realpath(dat_i)
            result["filter"] = os.path.realpath(filter_file)
            result["bias"] = os.path.realpath(bias_file)
            return result
        else:  # treat as single chip
            gticonfig(
                dat_json=dat_json,
                filter_file=filter_file,
                bias_file=bias_file,
                dat_out=dat_out,
                save_dir=save_dir
            )
            return {
                "dat0": os.path.realpath(dat_out),
                "filter": os.path.realpath(filter_file),
                "bias": os.path.realpath(bias_file)
            }


def convert_host_layers(
        checkpoint,
        host_layers,
        data_files,
        last_relu_var,
        reshape_to,
        save_dir,
        activation_bits=5
    ):
    reader = tf.train.NewCheckpointReader(checkpoint)
    for idx, layer in enumerate(host_layers):
        w = reader.get_tensor(layer.weight)
        b = reader.get_tensor(layer.bias)
        if idx == 0:
            relu_cap = reader.get_tensor(last_relu_var)
            gain = gti.chip.spec.get_max_activation(num_bits=activation_bits) / relu_cap
            w /= gain
            if reshape_to is not None:
                w = w.reshape(reshape_to)
                w = w.transpose(get_permute_axes("HWIO", "OIHW"))
                in_size = np.prod(reshape_to[:3])
                out_size = reshape_to[3]
                w = w.reshape((out_size, in_size))
            else:
                w = w.transpose(get_permute_axes("IO", "OI"))
        else:
            w = w.transpose(get_permute_axes("IO", "OI"))
        bin_path = os.path.join(save_dir, layer.name + ".bin")
        with open(bin_path, "wb") as f:
            out_size, in_size = w.shape
            np.array([in_size], dtype="<i").tofile(f)
            np.array([out_size], dtype="<i").tofile(f)
            w = np.asarray(w, order="C")
            w.tofile(f)
            b.tofile(f)
        data_files[layer.name] = os.path.realpath(bin_path)
    return data_files


def verify_paths(*paths):
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError("{} does not exist".format(p))


def concat_scopes(*scopes, sep="/"):
    result = [] 
    for s in scopes:  
        if not isinstance(s, str):  
            raise ValueError("Scope must be a string, but is {}".format(type(s)))  
        result += filter(None, s.split(sep)) 
    if not result: 
        return "" 
    return sep.join(result) + sep 


def update_model_json(model_json, data_files):
    """Update full MODEL JSON with newly generated data file paths, e.g. DAT, host layers, labels."""
    with open(model_json, "r+") as f:
        model_def = json.load(f)
        count_dat = 0
        for layer in model_def["layer"]:
            if layer["operation"] == "GTICNN":
                layer["data file"] = data_files["dat" + str(count_dat)] 
                count_dat += 1
            if layer["operation"] == "LABEL":
                layer["data file"] = data_files["label"]
            if layer["operation"] == "FC":
                layer["data file"] = data_files[layer["name"]]
        f.seek(0)
        json.dump(model_def, f, indent=4, sort_keys=True)
        f.truncate()


def update_dat_json(dat_json, new_shifts):
    """Update DAT JSON with newly calculated bit shifts/scaling factors from checkpoint."""
    with open(dat_json, "r+") as f:
        dat_json = json.load(f)
        cur_idx = 0
        for major_layer in dat_json["layer"]:
            old_shifts = major_layer["scaling"]
            num_old_shifts = len(old_shifts)
            updated_shifts = new_shifts[cur_idx : cur_idx + num_old_shifts]
            major_layer["scaling"] = [int(i) for i in updated_shifts]
            cur_idx += num_old_shifts
        f.seek(0)
        json.dump(dat_json, f, indent=4, sort_keys=True)
        f.truncate()
