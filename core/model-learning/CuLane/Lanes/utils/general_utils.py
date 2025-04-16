import tensorflow as tf
import os
import re

from tensorflow.keras import backend as K
from dataclasses import dataclass
from tensorflow.keras.utils import register_keras_serializable

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

@register_keras_serializable()
class MeanIoUMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes=4, name='mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(
            name='conf_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros',
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1)

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        mask = tf.not_equal(y_true, 0)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            num_classes=self.num_classes,
            dtype=tf.float32
        )

        self.confusion_matrix.assign_add(cm)

    def result(self):
        cm = self.confusion_matrix
        diag = tf.linalg.diag_part(cm)
        sum_rows = tf.reduce_sum(cm, axis=1)
        sum_cols = tf.reduce_sum(cm, axis=0)
        denominator = sum_rows + sum_cols - diag

        iou = tf.math.divide_no_nan(diag, denominator)

        valid = tf.where(sum_rows + sum_cols > 0)
        iou = tf.gather(iou, valid[:, 0])

        return tf.reduce_mean(iou)

    def reset_states(self):
        tf.keras.backend.set_value(self.confusion_matrix, tf.zeros_like(self.confusion_matrix))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes
        })
        return config

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    num_classes = tf.shape(y_pred)[-1]
    y_true = tf.cast(y_true, tf.int32)

    mask = tf.not_equal(y_true, 0)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(tf.argmax(y_pred, axis=-1), mask)

    y_true = tf.one_hot(y_true - 1, depth=num_classes - 1)
    y_pred = tf.one_hot(y_pred - 1, depth=num_classes - 1)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=0)
    union = tf.reduce_sum(y_true + y_pred, axis=0)

    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)

# def dice_coefficient(y_true, y_pred, smooth=1e-6):
#     num_classes = tf.shape(y_pred)[-1]
#     y_true = tf.cast(y_true, tf.int32)

#     y_true = tf.where(y_true != 0, y_true, tf.zeros_like(y_true))
#     y_true = tf.one_hot(y_true, depth=num_classes)
#     y_true = tf.cast(y_true, tf.float32)

#     y_pred = tf.nn.softmax(y_pred, axis=-1)

#     intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
#     union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])

#     dice = (2. * intersection + smooth) / (union + smooth)

#     return tf.reduce_mean(dice)

def get_tfrecord_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tfrecord')]

def resize_padding(image_tensor, resize_value):
    return tf.image.resize_with_pad(image_tensor, resize_value, resize_value, method=tf.image.ResizeMethod.BILINEAR)

def processing(config: DeepLabV3_config, proto):
    keys_to_features = {
        'image_path': tf.io.FixedLenFeature([], tf.string),
        'mask_raw': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        # 'channels': tf.io.FixedLenFeature([], tf.int64),
        'lane_quantity': tf.io.FixedLenFeature([], tf.int64)
    }

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    image_path = parsed_features['image_path']
    image_path = tf.strings.regex_replace(image_path, r'\\', '/')
    image_file = tf.io.read_file(tf.strings.join([str(config.base_path) + "/", image_path]))
    image = tf.image.decode_jpeg(image_file, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = resize_padding(image, config.input_resolution)

    mask = tf.io.decode_raw(parsed_features['mask_raw'], tf.uint8)
    mask = tf.cast(mask, tf.int32)
    mask = tf.reshape(mask, (config.input_resolution, config.input_resolution))
    mask.set_shape((config.input_resolution, config.input_resolution))

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