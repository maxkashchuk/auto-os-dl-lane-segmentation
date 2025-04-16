import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
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
    
    def get_config(self):
        config = super(ASPP, self).get_config()
        config.update({
            "filters": self.conv_layers[0].layers[0].filters,
            "dilation_rates": [layer.layers[0].dilation_rate for layer in self.conv_layers],
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def DeepLabV3(input_shape=(512, 512, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    
    backbone = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_tensor=inputs)

    output = backbone.get_layer("block13_sepconv2_bn").output

    aspp = ASPP(256, dilation_rates=[1, 6, 12, 18])(output)
    
    output = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax", dtype='float32')(aspp)

    output = tf.keras.layers.UpSampling2D(size=(input_shape[0] // output.shape[1], input_shape[1] // output.shape[2]))(output)
    
    output = tf.keras.layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(output)
    
    return tf.keras.Model(inputs, output)