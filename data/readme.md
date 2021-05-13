Data Pipeline Assumptions:
1. Images and labels have been converted to sharded TFRecord files using a particular encoding scheme. For example, use the following:
    - https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py
    - https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py
2. Data folders are arranged in the following way. The exact names do not matter as long as they are sharded TFRecord files, and split into sets, e.g. train/, val/, test/
    - dataset_name/train/shard_00001-of-00005.tfrecord
    - dataset_name/train/shard_00005-of-00005.tfrecord
    - dataset_name/val/shard_00001-of-00001.tfrecord