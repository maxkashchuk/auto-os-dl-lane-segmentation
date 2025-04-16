import os
import tensorflow as tf
import pandas as pd

from tensorflow.keras import mixed_precision
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from utils.general_utils import *
from deeplabv3 import *

EPOCHS = 5

MODEL_CHECKPOINT_PATH = 'deeplabv3_checkpoint.h5'

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

def deeplabv3_train():
    config = DeepLabV3_config(
        epochs = 5,
        input_resolution = 512,
        shuffle_size = 1024,
        batch_size = 2,
        model_checkpoint = MODEL_CHECKPOINT_PATH,
        base_path = BASE_PATH,
        base_mask_path = BASE_MASK_PATH,
        csv_path = CSV_PATH
    )

    train_dataset = load_dataset_from_shards(config, config.base_mask_path["train"]).repeat()

    validation_dataset = load_dataset_from_shards(config, config.base_mask_path["validation"])

    num_classes = pd.read_csv(config.csv_path)["max_lanes"].iloc[0]

    if os.path.exists(config.model_checkpoint):
        print("Loading existing model checkpoint...")
        model = load_model(config.model_checkpoint)
    else:
        print("Creating a new model...")

        model = DeepLabV3(input_shape=(config.input_resolution, config.input_resolution, 3), num_classes=num_classes)
        mean_iou_metric = MeanIoUMetric(num_classes=num_classes)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                metrics=[dice_coefficient, mean_iou_metric],
                steps_per_execution=5)

    checkpoint_cb = ModelCheckpoint("checkpoints/deeplabv3_checkpoint.keras", save_best_only=False, save_freq='epoch')

    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)

    model.fit(train_dataset, validation_data=validation_dataset, epochs=5, verbose=1, callbacks=[checkpoint_cb, early_stopping_cb], steps_per_epoch=39206)

    model.save('auto_os_road_lane_segmentation.keras')

def main():
    env_init()

    deeplabv3_train()

if __name__ == "__main__":
    main()