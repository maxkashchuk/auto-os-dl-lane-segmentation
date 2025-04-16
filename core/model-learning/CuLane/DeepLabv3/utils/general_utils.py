import tensorflow as tf
import os
import re

from tensorflow.keras import backend as K
from dataclasses import dataclass

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

class MeanIoUMetric(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, name='mean_iou', dtype=None):
        super().__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, depth=4)
    y_true = tf.cast(y_true, tf.float32)

    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2, 3])
    union = tf.reduce_sum(tf.abs(y_true) + tf.abs(y_pred), axis=[1, 2, 3])
    dice = 2. * intersection / (union + K.epsilon())
    return tf.reduce_mean(dice)

def get_tfrecord_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tfrecord')]

def resize_padding(image_tensor, resize_value):
    return tf.image.resize_with_pad(image_tensor, resize_value, resize_value,
                                    method=tf.image.ResizeMethod.BILINEAR)

def processing(config: DeepLabV3_config, proto):
    keys_to_features = {
        'image_path': tf.io.FixedLenFeature([], tf.string),
        'mask_raw': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channels': tf.io.FixedLenFeature([], tf.int64),
        'lane_quantity': tf.io.FixedLenFeature([], tf.int64)
    }

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    image_path = parsed_features['image_path']
    image_path = tf.strings.regex_replace(image_path, r'\\', '/')
    image_file = tf.io.read_file(tf.strings.join([str(config.base_path) + "/", image_path]))
    image = tf.image.decode_jpeg(image_file, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = resize_padding(image, config.input_resolution)

    height = parsed_features['height']
    width = parsed_features['width']
    channels = parsed_features['channels']
    mask_flat = tf.io.decode_raw(parsed_features['mask_raw'], tf.uint8)
    mask = tf.reshape(mask_flat, [height, width, channels])
    mask = resize_padding(mask, config.input_resolution)
    mask = mask[:, :, 0]
    mask = tf.cast(mask, tf.int32)

    return image, mask

def load_dataset_from_shards(config: DeepLabV3_config, directory, compression_type='ZLIB'):
    tfrecord_files = get_tfrecord_files(directory)

    dataset = tf.data.Dataset.list_files(tfrecord_files)

    dataset = dataset.interleave(
        lambda shard: tf.data.TFRecordDataset(shard, compression_type=compression_type, num_parallel_reads=tf.data.AUTOTUNE),
        cycle_length=tf.data.AUTOTUNE, 
        num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.map(lambda sample: processing(config, sample), num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset.shuffle(config.shuffle_size)
    
    dataset = dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def get_last_epoch(checkpoints_dir):
    checkpoint_files = os.listdir(checkpoints_dir)
    epoch_numbers = []

    pattern = re.compile(r'model_epoch_(\d+)\.keras')

    for filename in checkpoint_files:
        match = pattern.search(filename)
        if match:
            epoch_numbers.append(int(match.group(1)))

    if not epoch_numbers:
        return 0

    return max(epoch_numbers)