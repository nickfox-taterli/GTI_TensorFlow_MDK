# GTI TensorFlow Model Development Kit

## Terms and Conditions
You must agree to the terms and conditions in LICENCE/ directory before using this software.

## Dependencies
The kit has been tested with the following dependencies. To install missing Python dependencies, use pip. Anaconda/Miniconda and Docker are recommended for managing packages and environments. If using Docker, the official TensorFlow Docker 1.12 GPU image should work.
- Ubuntu 16.04
- CUDA 9.0
- CUDNN 7.3.1
- Python 3.6
- tensorflow-gpu 1.12
- numpy 1.15
- tqdm 4.28

## Directory Structure
|Directory   |Description
|------------|-----------
|/ (root)    |scripts to do various tasks, e.g. training, evaluation, conversion.
|checkpoints/|saved checkpoints
|data/       |datasets and samples
|gti/        |core libraries, e.g. models, layers, quantization, drivers.
|nets/       |configurations and intermediate files generated during conversion to chip format

## How to Run Task Scripts
Scripts provided in the root folder help perform various tasks, e.g. training, evaluation, conversion to chip. To run them, simply do:
```
python [TASK NAME].py
```
Some scripts may require and support many arguments. To see detailed descriptions of arguments, try:
```
python [TASK NAME].py --help
```

## Workflow Overview
1. Train floating-point model
2. Fine-tune with quantized convolutional layers
3. Fine-tune with quantized activation layers 
4. Fine-tune with batch norm and activation parameters fused into weights and biases
5. Convert to chip compatible format

Refer to user documentation for more details.