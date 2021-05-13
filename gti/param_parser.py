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

"""Argument parsers to parse training & evaluation parameters passed as script arguments."""
import argparse
from itertools import chain

import gti.chip.spec


class BaseParams(argparse.ArgumentParser):
    def __init__(self):
        super(BaseParams, self).__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.add_argument(
            "--net",
            type=str,
            required=True,
            help="network name, e.g. vgg16"
        ) 
        self.add_argument(
            "--chip",
            type=str,
            required=True,
            help="target chip"
        )
        self.add_argument(
            "--classes",
            type=int,
            default=1000,
            help="number of output classes"
        )
        self.add_argument(
            "--image_size",
            type=int,
            default=224,
            choices=gti.chip.spec.ALLOWED_IMAGE_SIZES,
            help="image size, where size == height == width"
        )
        

class QuantizationParams(BaseParams):
    def __init__(self):
        super(QuantizationParams, self).__init__()
        self.add_argument(
            "--quant_w",
            action="store_true",
            help="enable quantization of weights"
        ) 
        self.add_argument(
            "--quant_act",
            action="store_true",
            help="enable quantization of activations"
        )
        self.add_argument(
            "--fuse",
            action="store_true",
            help="fuse batch norm and ReLU cap into weight and bias to simulate chip operation during training"
        )


class TrainParser(QuantizationParams):
    def __init__(self):
        super(TrainParser, self).__init__()

        # Training 
        self.add_argument(
            "--train_data_dir",
            type=str,
            default="data/imagenet/train",
            help="directory where training images are stored",
        )
        self.add_argument(
            "--train_size",
            type=int,
            default=1281167,
            help="total number of images in training set"
        )
        self.add_argument(
            "--train_batch_size",
            type=int,
            default=64,
            help="number of images per training batch"
        )
        self.add_argument(
            "--learning_rate",
            type=float,
            default=1e-3,
            help="learning rate"
        )
        self.add_argument(
            "--weight_decay",
            type=float,
            default=1e-4,
            help="weight decay for L2 loss"
        )
        self.add_argument(
            "--num_epochs",
            type=int,
            default=100,
            help="number of epochs to train"
        )

        # Validation
        self.add_argument(
            "--val_data_dir",
            type=str,
            default="data/imagenet/val",
            help="directory where validation images are stored",
        )
        self.add_argument(
            "--val_size",
            type=int,
            default=50000,
            help="total number of images in validation set"
        )
        self.add_argument(
            "--val_batch_size",
            type=int,
            default=100,
            help="number of images per validation batch"
        )

        # Checkpoints & weights
        self.add_argument(
            "--best_checkpoint_dir",
            type=str,
            default="checkpoints/best",
            help="directory to save best checkpoint"
        )
        self.add_argument(
            "--last_checkpoint_dir",
            type=str,
            default="checkpoints/last",
            help="directory to save latest N checkpoint(s)"
        )
        self.add_argument(
            "--keep_last_n_checkpoints",
            type=int,
            default=5,
            help="how many latest checkpoints to keep"
        )
        self.add_argument(
            "--resume_from",
            type=str,
            default=None,
            help="checkpoint to resume training from"
        )
        self.add_argument(
            "--num_gpus",
            type=int,
            default=1,
            help="number of GPUs to use for training"
        )


class EvalParser(QuantizationParams):
    def __init__(self):
        super(EvalParser, self).__init__()
        self.add_argument(
            "--data_dir",
            type=str,
            default="data/imagenet/val",
            help="dataset directory",
        )
        self.add_argument(
            "--data_size",
            type=int,
            default=50000,
            help="total number of images in dataset"
        )
        self.add_argument(
            "--batch_size",
            type=int,
            default=100,
            help="number of images per batch"
        )
        self.add_argument(
            "--checkpoint",
            type=str,
            required=True,
            help="checkpoint to evaluate"
        )
        self.add_argument(
            "--num_gpus",
            type=int,
            default=1,
            help="number of GPUs to use for evaluation"
        )


class CheckpointInitParser(BaseParams):
    def __init__(self):
        super(CheckpointInitParser, self).__init__()
        self.add_argument(
            "--npy",
            type=str,
            required=True,
            help="numpy file to initialize TensorFlow checkpoint from"
        )
        self.add_argument(
            "--save_checkpoint_to",
            type=str,
            default="checkpoints/last",
            help="directory to save checkpoint"
        )


class ReLUCapParser(QuantizationParams):
    def __init__(self):
        super(ReLUCapParser, self).__init__()
        self.add_argument(
            "--data_dir",
            type=str,
            default="data/imagenet/train",
            help="dataset directory",
        )
        self.add_argument(
            "--data_size",
            type=int,
            default=1281167,
            help="total number of images in dataset"
        )
        self.add_argument(
            "--batch_size",
            type=int,
            default=64,
            help="number of images per batch"
        )
        self.add_argument(
            "--percentile",
            type=float,
            default=99.99,
            help="percentile to sample ReLU outputs at"
        )
        self.add_argument(
            "--num_batches",
            type=int,
            default=5,
            help="how many batches to sample ReLU outputs for"
        )
        self.add_argument(
            "--checkpoint",
            type=str,
            required=True,
            help="checkpoint used to compute ReLU outputs"
        )


class ConversionParser(argparse.ArgumentParser):
    def __init__(self):
        super(ConversionParser, self).__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.add_argument(
            "--net",
            type=str,
            required=True,
            help="network name, e.g. vgg16"
        ) 
        self.add_argument(
            "--chip",
            type=str,
            required=True,
            help="target chip"
        )
        self.add_argument(
            "--checkpoint",
            type=str,
            required=True,
            help="checkpoint path",
        )
        self.add_argument(
            "--ten_bit_act",
            action="store_true",
            help="whether checkpoint has been trained with 10-bit activation for last chip layer"
        )
        self.add_argument(
            "--fuse",
            action="store_true",
            help="whether batch norms and ReLU caps in checkpoint have already been fused into weights and biases"
        )


class ChipInferParser(argparse.ArgumentParser):
    def __init__(self):
        super(ChipInferParser, self).__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.add_argument(
            "--net",
            type=str,
            required=True,
            help="network name, e.g. vgg16"
        ) 
        self.add_argument(
            "--chip",
            type=str,
            required=True,
            help="chip"
        )
        self.add_argument(
            "--checkpoint",
            type=str,
            required=True,
            help="checkpoint path. Must be the same checkpoint used in model conversion."
        )
        self.add_argument(
            "--last_relu_cap",
            type=str,
            required=True,
            help="variable name of the last ReLU cap variable on-chip as saved in checkpoint, e.g. vgg16/conv5_3/relu_cap"
        )
        self.add_argument(
            "--ten_bit_act",
            action="store_true",
            help="whether checkpoint has been trained with 10-bit activation for last chip layer"
        )
        self.add_argument(
            "--classes",
            type=int,
            default=1000,
            help="number of classes" 
        )
        self.add_argument(
            "--data_dir",
            type=str,
            default="data/imagenet/val",
            help="dataset directory for evaluation on chip"
        )
        self.add_argument(
            "--data_size",
            type=int,
            default=50000,
            help="dataset size" 
        )
        self.add_argument(
            "--image_size",
            type=int,
            default=224,
            choices=gti.chip.spec.ALLOWED_IMAGE_SIZES,
            help="image size, where size == height == width"
        )
        