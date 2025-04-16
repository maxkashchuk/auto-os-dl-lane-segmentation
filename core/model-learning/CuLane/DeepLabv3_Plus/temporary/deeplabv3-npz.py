import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import pandas as pd
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

INPUT_RESOLUTION = 512

BASE_PATH = r"../../../../"

BASE_MASK_PATH = {"train": f"{BASE_PATH}dataset-description/train",
                   "validation": f"{BASE_PATH}dataset-description/validation",
                   "test": f"{BASE_PATH}dataset-description/test"}

CSV_PATH_TRAIN = BASE_PATH + r"dataset-description/train.csv"
CSV_PATH_VALIDATION = BASE_PATH + r"dataset-description/validation.csv"
CSV_PATH_TEST = BASE_PATH + r"dataset-description/test.csv"

def resize_padding(image_tensor, resize_value):
    return tf.image.resize_with_pad(image_tensor, resize_value, resize_value,
                                    method=tf.image.ResizeMethod.BILINEAR)

def num_classes_calculation(*lane_quantity_frames):
    return max(2, max([frame.max() for frame in lane_quantity_frames]) + 1)

def image_load(image_path, resize_value=INPUT_RESOLUTION):
    image_path = tf.strings.regex_replace(image_path, "\\\\", "/")
    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = resize_padding(image, resize_value)

    return image

def mask_load(mask_path, resize_value=INPUT_RESOLUTION):
    mask_path = mask_path.numpy().decode("utf-8").replace("\\", "/")

    mask = np.load(mask_path)['arr_0']

    mask = resize_padding(mask, resize_value).numpy().astype(np.uint8)

    return mask[:, :, 0]

def load_data(image_path, mask_path):
    image = image_load(image_path)

    mask = tf.py_function(func=mask_load, inp=[mask_path], Tout=tf.uint8)

    mask.set_shape([INPUT_RESOLUTION, INPUT_RESOLUTION])

    return image, mask


def dataset_formation(images, masks, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))

    dataset = dataset.map(lambda image, mask: (load_data(image, mask)))

    # dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)

    return dataset

class ASPP(tf.keras.layers.Layer):
    def __init__(self, filters, dilation_rates):
        super(ASPP, self).__init__()
        self.conv_layers = []
        for rate in dilation_rates:
            self.conv_layers.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters, 3, padding="same", dilation_rate=rate, use_bias=False),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU()
                ])
            )
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.conv1x1 = tf.keras.layers.Conv2D(filters, 1, activation='relu')

    def call(self, inputs):
        outputs = [conv(inputs) for conv in self.conv_layers]
        avg_pooled = self.global_avg_pool(inputs)
        avg_pooled = tf.expand_dims(tf.expand_dims(avg_pooled, 1), 1)
        avg_pooled = self.conv1x1(avg_pooled)
        avg_pooled = tf.keras.layers.UpSampling2D(size=(inputs.shape[1], inputs.shape[2]))(avg_pooled)
        return tf.concat(outputs + [avg_pooled], axis=-1)

def DeepLabV3(input_shape=(512, 512, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    
    backbone = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_tensor=inputs)
    output = backbone.get_layer("block13_sepconv2_bn").output

    aspp = ASPP(256, dilation_rates=[1, 6, 12, 18])(output)
    
    output = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax")(aspp)
    
    output = tf.keras.layers.UpSampling2D(size=(input_shape[0] // output.shape[1], input_shape[1] // output.shape[2]))(output)
    
    return tf.keras.Model(inputs, output)

df_train = pd.read_csv(CSV_PATH_TRAIN)

df_test = pd.read_csv(CSV_PATH_TEST)

df_validation = pd.read_csv(CSV_PATH_VALIDATION)

num_classes_calculated = num_classes_calculation(df_train['lane_quantity'].astype(int),
                                                 df_test['lane_quantity'].astype(int),
                                                 df_validation['lane_quantity'].astype(int))

train_dataset = dataset_formation(df_train['image_path'].apply(lambda x: BASE_PATH + x),
                                  df_train['tensor_path'].apply(lambda x: BASE_MASK_PATH['train'] + x),
                                  batch_size=1)

# train_dataset = train_dataset.shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE)

validation_dataset = dataset_formation(df_validation['image_path'].apply(lambda x: BASE_PATH + x),
                                  df_validation['tensor_path'].apply(lambda x: BASE_MASK_PATH['validation'] + x),
                                  batch_size=1)

# validation_dataset = validation_dataset.shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE)

model = DeepLabV3(input_shape=(INPUT_RESOLUTION, INPUT_RESOLUTION, 3), num_classes=num_classes_calculated)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])

model.fit(train_dataset, validation_data=validation_dataset, epochs=10, verbose=1)

model.save('auto_os_road_lane_segmentation.h5')