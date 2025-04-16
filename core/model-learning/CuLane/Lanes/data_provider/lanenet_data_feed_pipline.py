#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-23 下午3:54
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_data_feed_pipline.py
# @IDE: PyCharm
"""
Lanenet data feed pip line
"""
import time
import glob
import os
import os.path as ops
import random

import numpy as np
import tensorflow as tf
import loguru

from local_utils.config_utils import parse_config_utils
# from data_provider import tf_io_pipline_tools
from dataclasses import dataclass
from typing import List

CFG = parse_config_utils.lanenet_cfg
LOG = loguru.logger

@dataclass
class DeepLabV3_config:
    epochs: int
    input_resolution: int
    shuffle_size: int
    batch_size: int
    model_checkpoint: str
    base_path: str
    base_mask_path: str
    csv_path: str

def sample_points(xs, ys, num_points=20):
    # Рівномірно вибираємо num_points з усіх точок (якщо є більше)
    if len(xs) < num_points:
        # Додати нулі, якщо точок менше
        xs = np.pad(xs, (0, num_points - len(xs)), mode='constant')
        ys = np.pad(ys, (0, num_points - len(ys)), mode='constant')
    else:
        indices = np.linspace(0, len(xs) - 1, num_points).astype(int)
        xs = xs[indices]
        ys = ys[indices]
    return np.stack([xs, ys], axis=1).astype(np.float32)  # shape (num_points, 2)

def mask_to_coords(instance_mask, num_lanes=4, num_points=20):
    """
    instance_mask: np.array shape (H, W) з індексами ліній
    Повертає coords shape (num_lanes, num_points, 2)
    """
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids != 0]  # пропускаємо фон

    coords_list = []
    for lane_id in unique_ids:
        ys, xs = np.where(instance_mask == lane_id)
        if len(xs) == 0:
            continue
        # Сортуємо точки за y (висотою)
        sort_idx = np.argsort(ys)
        xs, ys = xs[sort_idx], ys[sort_idx]

        points = sample_points(xs, ys, num_points)
        coords_list.append(points)

    # Якщо ліній менше ніж num_lanes, додаємо пусті
    while len(coords_list) < num_lanes:
        coords_list.append(np.zeros((num_points, 2), dtype=np.float32))

    coords = np.stack(coords_list[:num_lanes], axis=0)  # (num_lanes, num_points, 2)
    return coords

def get_tfrecord_files(directory: str) -> List[str]:
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tfrecord')]


def resize_padding(image_tensor: tf.Tensor, resize_value: int) -> tf.Tensor:
    return tf.image.resize_with_pad(
        image_tensor, resize_value, resize_value, method=tf.image.ResizeMethod.BILINEAR
    )

def get_tfrecord_files(directory: str) -> List[str]:
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tfrecord')]


def resize_padding(image_tensor: tf.Tensor, resize_value: int) -> tf.Tensor:
    return tf.image.resize_with_pad(image_tensor, resize_value, resize_value, method=tf.image.ResizeMethod.BILINEAR)


def parse_example(proto: tf.Tensor, config: DeepLabV3_config) -> tuple:
    features = {
        'image_path': tf.io.FixedLenFeature([], tf.string),
        'mask_raw': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        # 'lane_quantity': tf.io.FixedLenFeature([], tf.int64),
    }

    parsed = tf.io.parse_single_example(proto, features)

    # Формуємо шлях до зображення
    image_path = tf.strings.regex_replace(parsed['image_path'], r'\\', '/')
    full_path = tf.strings.join([config.base_path, "/", image_path])
    image_data = tf.io.read_file(full_path)
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = resize_padding(image, config.input_resolution)

    # Обробка маски
    instance_mask = tf.io.decode_raw(parsed['mask_raw'], tf.uint8)
    instance_mask = tf.cast(instance_mask, tf.int32)
    instance_mask = tf.reshape(instance_mask, (config.input_resolution, config.input_resolution))

    coords = tf_mask_to_coords(instance_mask)

    return image, coords


def create_dataset(config: DeepLabV3_config, flags: str = 'train') -> tf.data.Dataset:
    flags = flags.lower()
    if flags not in ['train', 'val']:
        raise ValueError("flags must be either 'train' or 'val'")

    tfrecord_dir = ops.join(config.base_path, 'dataset-description', flags)
    tfrecord_files = get_tfrecord_files(tfrecord_dir)
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found in {tfrecord_dir}")

    dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    dataset = dataset.interleave(
        lambda path: tf.data.TFRecordDataset(path, compression_type='ZLIB'),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(lambda x: parse_example_with_coords(x, config), num_parallel_calls=tf.data.AUTOTUNE)

    if flags == 'train':
        dataset = dataset.shuffle(config.shuffle_size)

    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# class LaneNetDataProducer(object):
#     """
#     Convert raw image file into tfrecords
#     """

#     def __init__(self):
#         """

#         """
#         self._dataset_dir = CFG.DATASET.DATA_DIR
#         self._tfrecords_save_dir = ops.join(self._dataset_dir, 'tfrecords')
#         self._train_example_index_file_path = CFG.DATASET.TRAIN_FILE_LIST
#         self._test_example_index_file_path = CFG.DATASET.TEST_FILE_LIST
#         self._val_example_index_file_path = CFG.DATASET.VAL_FILE_LIST

#         self._gt_image_dir = ops.join(self._dataset_dir, 'gt_image')
#         self._gt_binary_image_dir = ops.join(self._dataset_dir, 'gt_binary_image')
#         self._gt_instance_image_dir = ops.join(self._dataset_dir, 'gt_instance_image')

#         if not self._is_source_data_complete():
#             raise ValueError('Source image data is not complete, '
#                              'please check if one of the image folder does not exist')

#         if not self._is_training_sample_index_file_complete():
#             self._generate_training_example_index_file()

#     def generate_tfrecords(self):
#         """
#         Generate tensorflow records file
#         :return:
#         """

#         def _read_training_example_index_file(_index_file_path):

#             assert ops.exists(_index_file_path)

#             _example_gt_path_info = []
#             _example_gt_binary_path_info = []
#             _example_gt_instance_path_info = []

#             with open(_index_file_path, 'r') as _file:
#                 for _line in _file:
#                     _example_info = _line.rstrip('\r').rstrip('\n').split(' ')
#                     _example_gt_path_info.append(_example_info[0])
#                     _example_gt_binary_path_info.append(_example_info[1])
#                     _example_gt_instance_path_info.append(_example_info[2])

#             ret = {
#                 'gt_path_info': _example_gt_path_info,
#                 'gt_binary_path_info': _example_gt_binary_path_info,
#                 'gt_instance_path_info': _example_gt_instance_path_info
#             }

#             return ret

#         # make save dirs
#         os.makedirs(self._tfrecords_save_dir, exist_ok=True)

#         # start generating training example tfrecords
#         LOG.info('Start generating training example tfrecords')

#         # collecting train images paths info
#         train_image_paths_info = _read_training_example_index_file(self._train_example_index_file_path)
#         train_gt_images_paths = train_image_paths_info['gt_path_info']
#         train_gt_binary_images_paths = train_image_paths_info['gt_binary_path_info']
#         train_gt_instance_images_paths = train_image_paths_info['gt_instance_path_info']
#         train_tfrecords_paths = ops.join(self._tfrecords_save_dir, 'tusimple_train.tfrecords')
#         tf_io_pipline_tools.write_example_tfrecords(
#             train_gt_images_paths,
#             train_gt_binary_images_paths,
#             train_gt_instance_images_paths,
#             train_tfrecords_paths
#         )
#         LOG.info('Generating training example tfrecords complete')

#         # start generating validation example tfrecords
#         LOG.info('Start generating validation example tfrecords')

#         # collecting validation images paths info
#         val_image_paths_info = _read_training_example_index_file(self._val_example_index_file_path)
#         val_gt_images_paths = val_image_paths_info['gt_path_info']
#         val_gt_binary_images_paths = val_image_paths_info['gt_binary_path_info']
#         val_gt_instance_images_paths = val_image_paths_info['gt_instance_path_info']
#         val_tfrecords_paths = ops.join(self._tfrecords_save_dir, 'tusimple_val.tfrecords')
#         tf_io_pipline_tools.write_example_tfrecords(
#             val_gt_images_paths,
#             val_gt_binary_images_paths,
#             val_gt_instance_images_paths,
#             val_tfrecords_paths
#         )
#         LOG.info('Generating validation example tfrecords complete')

#         # generate test example tfrecords
#         LOG.info('Start generating testing example tfrecords')

#         # collecting test images paths info
#         test_image_paths_info = _read_training_example_index_file(self._test_example_index_file_path)
#         test_gt_images_paths = test_image_paths_info['gt_path_info']
#         test_gt_binary_images_paths = test_image_paths_info['gt_binary_path_info']
#         test_gt_instance_images_paths = test_image_paths_info['gt_instance_path_info']
#         test_tfrecords_paths = ops.join(self._tfrecords_save_dir, 'tusimple_test.tfrecords')
        # tf_io_pipline_tools.write_example_tfrecords(
        #     test_gt_images_paths,
        #     test_gt_binary_images_paths,
        #     test_gt_instance_images_paths,
        #     test_tfrecords_paths
        # )
#         LOG.info('Generating testing example tfrecords complete')

#         return

#     def _is_source_data_complete(self):
#         """
#         Check if source data complete
#         :return:
#         """
#         return \
#             ops.exists(self._gt_binary_image_dir) and \
#             ops.exists(self._gt_instance_image_dir) and \
#             ops.exists(self._gt_image_dir)

#     def _is_training_sample_index_file_complete(self):
#         """
#         Check if the training sample index file is complete
#         :return:
#         """
#         return \
#             ops.exists(self._train_example_index_file_path) and \
#             ops.exists(self._test_example_index_file_path) and \
#             ops.exists(self._val_example_index_file_path)

#     def _generate_training_example_index_file(self):
#         """
#         Generate training example index file, split source file into 0.85, 0.1, 0.05 for training,
#         testing and validation. Each image folder are processed separately
#         :return:
#         """

#         def _gather_example_info():
#             """

#             :return:
#             """
#             _info = []

#             for _gt_image_path in glob.glob('{:s}/*.png'.format(self._gt_image_dir)):
#                 _gt_binary_image_name = ops.split(_gt_image_path)[1]
#                 _gt_binary_image_path = ops.join(self._gt_binary_image_dir, _gt_binary_image_name)
#                 _gt_instance_image_name = ops.split(_gt_image_path)[1]
#                 _gt_instance_image_path = ops.join(self._gt_instance_image_dir, _gt_instance_image_name)

#                 assert ops.exists(_gt_binary_image_path), '{:s} not exist'.format(_gt_binary_image_path)
#                 assert ops.exists(_gt_instance_image_path), '{:s} not exist'.format(_gt_instance_image_path)

#                 _info.append('{:s} {:s} {:s}\n'.format(
#                     _gt_image_path,
#                     _gt_binary_image_path,
#                     _gt_instance_image_path)
#                 )

#             return _info

#         def _split_training_examples(_example_info):
#             random.shuffle(_example_info)

#             _example_nums = len(_example_info)

#             _train_example_info = _example_info[:int(_example_nums * 0.85)]
#             _val_example_info = _example_info[int(_example_nums * 0.85):int(_example_nums * 0.9)]
#             _test_example_info = _example_info[int(_example_nums * 0.9):]

#             return _train_example_info, _test_example_info, _val_example_info

#         train_example_info, test_example_info, val_example_info = _split_training_examples(_gather_example_info())

#         random.shuffle(train_example_info)
#         random.shuffle(test_example_info)
#         random.shuffle(val_example_info)

#         with open(ops.join(self._dataset_dir, 'train.txt'), 'w') as file:
#             file.write(''.join(train_example_info))

#         with open(ops.join(self._dataset_dir, 'test.txt'), 'w') as file:
#             file.write(''.join(test_example_info))

#         with open(ops.join(self._dataset_dir, 'val.txt'), 'w') as file:
#             file.write(''.join(val_example_info))

#         LOG.info('Generating training example index file complete')

#         return


# class LaneNetDataFeeder(object):
#     """
#     Read training examples from tfrecords for nsfw model
#     """

#     def __init__(self, flags='train'):
#         """

#         :param flags:
#         """
#         self._dataset_dir = CFG.DATASET.DATA_DIR
#         self._epoch_nums = CFG.TRAIN.EPOCH_NUMS
#         self._train_batch_size = CFG.TRAIN.BATCH_SIZE
#         self._val_batch_size = CFG.TRAIN.VAL_BATCH_SIZE

#         self._tfrecords_dir = ops.join(self._dataset_dir, 'tfrecords')
#         if not ops.exists(self._tfrecords_dir):
#             raise ValueError('{:s} not exist, please check again'.format(self._tfrecords_dir))

#         self._dataset_flags = flags.lower()
#         if self._dataset_flags not in ['train', 'val']:
#             raise ValueError('flags of the data feeder should be \'train\', \'val\'')

#     def __len__(self):
#         """

#         :return:
#         """
#         tfrecords_file_paths = ops.join(self._tfrecords_dir, 'tusimple_{:s}.tfrecords'.format(self._dataset_flags))
#         assert ops.exists(tfrecords_file_paths), '{:s} not exist'.format(tfrecords_file_paths)

#         sample_counts = 0
#         sample_counts += sum(1 for _ in tf.python_io.tf_record_iterator(tfrecords_file_paths))
#         if self._dataset_flags == 'train':
#             num_batchs = int(np.ceil(sample_counts / self._train_batch_size))
#         elif self._dataset_flags == 'val':
#             num_batchs = int(np.ceil(sample_counts / self._val_batch_size))
#         else:
#             raise ValueError('Wrong dataset flags')
#         return num_batchs

#     def next_batch(self, batch_size):
#         """
#         dataset feed pipline input
#         :param batch_size:
#         :return: A tuple (images, labels), where:
#                     * images is a float tensor with shape [batch_size, H, W, C]
#                       in the range [-0.5, 0.5].
#                     * labels is an int32 tensor with shape [batch_size] with the true label,
#                       a number in the range [0, CLASS_NUMS).
#         """
#         tfrecords_file_paths = ops.join(self._tfrecords_dir, 'tusimple_{:s}.tfrecords'.format(self._dataset_flags))
#         assert ops.exists(tfrecords_file_paths), '{:s} not exist'.format(tfrecords_file_paths)

#         with tf.device('/cpu:0'):
#             with tf.name_scope('input_tensor'):

#                 # TFRecordDataset opens a binary file and reads one record at a time.
#                 # `tfrecords_file_paths` could also be a list of filenames, which will be read in order.
#                 dataset = tf.data.TFRecordDataset(tfrecords_file_paths)

#                 # The map transformation takes a function and applies it to every element
#                 # of the dataset.
#                 dataset = dataset.map(
#                     map_func=tf_io_pipline_tools.decode,
#                     num_parallel_calls=CFG.DATASET.CPU_MULTI_PROCESS_NUMS
#                 )
#                 if self._dataset_flags == 'train':
#                     dataset = dataset.map(
#                         map_func=tf_io_pipline_tools.augment_for_train,
#                         num_parallel_calls=CFG.DATASET.CPU_MULTI_PROCESS_NUMS
#                     )
#                 elif self._dataset_flags == 'val':
#                     dataset = dataset.map(
#                         map_func=tf_io_pipline_tools.augment_for_test,
#                         num_parallel_calls=CFG.DATASET.CPU_MULTI_PROCESS_NUMS
#                     )
#                 dataset = dataset.map(
#                     map_func=tf_io_pipline_tools.normalize,
#                     num_parallel_calls=CFG.DATASET.CPU_MULTI_PROCESS_NUMS
#                 )

#                 dataset = dataset.shuffle(buffer_size=512)
#                 # repeat num epochs
#                 dataset = dataset.repeat(self._epoch_nums)

#                 dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
#                 dataset = dataset.prefetch(buffer_size=128)

#                 iterator = dataset.make_one_shot_iterator()

#         return iterator.get_next(name='{:s}_IteratorGetNext'.format(self._dataset_flags))


# if __name__ == '__main__':
#     """
#     test code
#     """
#     train_dataset = LaneNetDataFeeder(flags='train')

#     src_images, binary_label_images, instance_label_images = train_dataset.next_batch(batch_size=8)

#     count = 1
#     with tf.Session() as sess:
#         while True:
#             try:
#                 t_start = time.time()
#                 images, binary_labels, instance_labels = sess.run(
#                     [src_images, binary_label_images, instance_label_images]
#                 )
#                 print('Iter: {:d}, cost time: {:.5f}s'.format(count, time.time() - t_start))
#                 count += 1
#                 src_image = np.array((images[0] + 1.0) * 127.5, dtype=np.uint8)
#             except tf.errors.OutOfRangeError as err:
#                 print(err)
#                 raise err
