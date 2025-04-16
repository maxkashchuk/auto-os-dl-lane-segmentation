from tensorflow.keras import layers, models
from data_provider.lanenet_data_feed_pipline import *

EPOCHS = 5

MODEL_CHECKPOINT_PATH = 'checkpoints/deeplabv3_plus_checkpoint.keras'

INPUT_RESOLUTION = 512

BASE_PATH = os.path.join("..", "..", "..", "..")

BASE_MASK_PATH = {"train": os.path.join(BASE_PATH, "dataset-description", "train"),
                   "validation": os.path.join(BASE_PATH, "dataset-description", "validation"),
                   "test": os.path.join(BASE_PATH, "dataset-description", "test")}

CSV_PATH = os.path.join(BASE_PATH, "dataset-description", "summary.csv")

def build_lane_regression_model(input_shape=(512, 512, 3), num_points=20, num_lanes=4):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)

    outputs = layers.Dense(num_lanes * num_points * 2)(x)  # x, y for each point
    outputs = layers.Reshape((num_lanes, num_points, 2))(outputs)

    return models.Model(inputs, outputs)

config = DeepLabV3_config(
            epochs = 40,
            input_resolution = 312,
            shuffle_size = 1024,
            batch_size = 2,
            model_checkpoint = MODEL_CHECKPOINT_PATH,
            base_path = BASE_PATH,
            base_mask_path = BASE_MASK_PATH,
            csv_path = CSV_PATH
        )

model = build_lane_regression_model()
model.compile(optimizer='adam', loss='mse')

train_dataset = create_dataset(config, flags='train')
val_dataset = create_dataset(config, flags='val')

model.fit(train_dataset, epochs=50, validation_data=val_dataset)