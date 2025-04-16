import os
import tensorflow as tf
import pandas as pd

from tensorflow.keras.applications import Xception, ResNet50
from tensorflow.keras import mixed_precision
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.utils import register_keras_serializable

from utils.general_utils import *
from deeplabv3_plus import DeeplabV3Plus

BACKBONES = {
    'resnet50': {
        'model': ResNet50,
        'feature_1': 'conv4_block6_out',  # high-level features
        'feature_2': 'conv2_block3_out'   # low-level features
    },
    'xception': {
        'model': Xception,
        'feature_1': 'block13_sepconv2_bn',  # high-level features
        'feature_2': 'block4_sepconv2_bn'    # low-level features
    }
}

EPOCHS = 5

MODEL_CHECKPOINT_PATH = 'checkpoints/deeplabv3_plus_checkpoint.keras'

INPUT_RESOLUTION = 512

BASE_PATH = os.path.join("..", "..", "..", "..")

BASE_MASK_PATH = {"train": os.path.join(BASE_PATH, "dataset-description", "train"),
                   "validation": os.path.join(BASE_PATH, "dataset-description", "validation"),
                   "test": os.path.join(BASE_PATH, "dataset-description", "test")}

CSV_PATH = os.path.join(BASE_PATH, "dataset-description", "summary.csv")

def env_init():
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    tf.keras.backend.clear_session()

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    tf.config.optimizer.set_jit(True)

@register_keras_serializable()
def masked_sparse_categorical_crossentropy(y_true, y_pred):
    mask = tf.not_equal(y_true, 255)
    
    y_true_clean = tf.where(mask, y_true, tf.zeros_like(y_true))

    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_clean, y_pred, from_logits=False, ignore_class=255)

    mask = tf.cast(mask, loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def create_sample_weights(y_true):
    # y_true shape: (batch_size, height, width)
    # 255 - padding → weight=0, інші - 1
    weights = tf.cast(tf.not_equal(y_true, 255), tf.float32)
    return weights

def add_sample_weights_to_dataset(dataset):
    def map_fn(image, label):
        weights = create_sample_weights(label)
        return (image, label), weights
    return dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

def deeplabv3_train():
    config = DeepLabV3_config(
        epochs = 40,
        input_resolution = 512,
        shuffle_size = 1024,
        batch_size = 2,
        model_checkpoint = MODEL_CHECKPOINT_PATH,
        base_path = BASE_PATH,
        base_mask_path = BASE_MASK_PATH,
        csv_path = CSV_PATH
    )

    train_dataset = load_dataset_from_shards(config, config.base_mask_path["train"]).repeat()
    train_dataset = add_sample_weights_to_dataset(train_dataset)

    validation_dataset = load_dataset_from_shards(config, config.base_mask_path["validation"])
    validation_dataset = add_sample_weights_to_dataset(validation_dataset)

    num_classes = pd.read_csv(config.csv_path)["max_lanes"].iloc[0]

    if os.path.exists(config.model_checkpoint):
        print("Loading existing model checkpoint...")
        get_custom_objects().update({'DeepLabV3Plus': DeeplabV3Plus(num_classes=4)})
        custom_objects = {
            "DeeplabV3Plus": DeeplabV3Plus,
            "dice_coefficient": dice_coefficient,
            "MeanIoUMetric": MeanIoUMetric(num_classes=4)
        }
        model = load_model(config.model_checkpoint, custom_objects=custom_objects)

        model.load_weights(config.model_checkpoint)

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-4,
            decay_steps=39206 * config.epochs,
            alpha=0.1
        )

        mean_iou_metric = MeanIoUMetric(num_classes=num_classes)

        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4), 
                loss=masked_sparse_categorical_crossentropy, 
                metrics=[dice_coefficient, mean_iou_metric],
                steps_per_execution=5)
    else:
        print("Creating a new model...")

        model = DeeplabV3Plus(num_classes=num_classes+1, backbone='xception')
        mean_iou_metric = MeanIoUMetric(num_classes=num_classes+1)

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-4,
            decay_steps=39206 * config.epochs,
            alpha=0.1
        )

        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4), 
                loss=masked_sparse_categorical_crossentropy, 
                metrics=[dice_coefficient, mean_iou_metric],
                steps_per_execution=5)


    checkpoint_cb = ModelCheckpoint(config.model_checkpoint, save_best_only=True, monitor="val_loss", mode="min", save_freq='epoch')

    early_stopping_cb = EarlyStopping(monitor='val_dice_coefficient', patience=12, mode='max', restore_best_weights=True)

    reduce_lr_cb = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )

    tensorboard_cb = TensorBoard(
        log_dir="logs/fit",
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )

    model.fit(train_dataset, validation_data=validation_dataset, epochs=config.epochs, verbose=1, callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb, tensorboard_cb], steps_per_epoch=39206)

    # model.fit(train_dataset, validation_data=validation_dataset, epochs=config.epochs, verbose=1, callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb, tensorboard_cb], steps_per_epoch=39206)

    model.save('auto_os_road_lane_segmentation.keras')

def main():
    env_init()

    deeplabv3_train()

if __name__ == "__main__":
    main()