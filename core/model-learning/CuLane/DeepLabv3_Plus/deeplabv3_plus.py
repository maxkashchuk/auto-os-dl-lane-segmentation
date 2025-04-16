import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import register_keras_serializable

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, kernel_size, padding, dilation_rate,
                 kernel_initializer, use_bias, conv_activation=None):
        super(ConvBlock, self).__init__()

        self.conv = tf.keras.layers.Conv2D(
            n_filters, kernel_size=kernel_size, padding=padding,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias, dilation_rate=dilation_rate,
            activation=conv_activation)

        self.batch_norm = tf.keras.layers.BatchNormalization(dtype='float32')

    def call(self, inputs, **kwargs):
        tensor = self.conv(inputs)
        tensor = self.batch_norm(tensor)
        tensor = tf.nn.relu(tensor)
        return tensor


class AtrousSpatialPyramidPooling(tf.keras.layers.Layer):
    def __init__(self):
        super(AtrousSpatialPyramidPooling, self).__init__()

        self.avg_pool = None
        self.conv1, self.conv2 = None, None
        self.pool = None
        self.out1, self.out6, self.out12, self.out18 = None, None, None, None

    @staticmethod
    def _get_conv_block(kernel_size, dilation_rate, use_bias=False):
        return ConvBlock(256,
                         kernel_size=kernel_size,
                         dilation_rate=dilation_rate,
                         padding='same',
                         use_bias=use_bias,
                         kernel_initializer=tf.keras.initializers.he_normal())

    def build(self, input_shape):
        dummy_tensor = tf.random.normal((1, *input_shape[1:]))

        self.avg_pool = tf.keras.layers.AveragePooling2D(
            pool_size=(input_shape[-3], input_shape[-2]))

        self.conv1 = AtrousSpatialPyramidPooling._get_conv_block(
            kernel_size=1, dilation_rate=1, use_bias=True)

        self.conv2 = AtrousSpatialPyramidPooling._get_conv_block(
            kernel_size=1, dilation_rate=1)

        dummy_tensor = self.conv1(self.avg_pool(dummy_tensor))

        self.pool = tf.keras.layers.UpSampling2D(
            size=(
                input_shape[-3] // dummy_tensor.shape[1],
                input_shape[-2] // dummy_tensor.shape[2]
            ),
            interpolation='bilinear'
        )

        self.out1, self.out6, self.out12, self.out18 = map(
            lambda tup: AtrousSpatialPyramidPooling._get_conv_block(
                kernel_size=tup[0], dilation_rate=tup[1]
            ),
            [(1, 1), (3, 6), (3, 12), (3, 18)]
        )

    def call(self, inputs, **kwargs):
        tensor = self.avg_pool(inputs)
        tensor = self.conv1(tensor)
        tensor = tf.keras.layers.Concatenate(axis=-1)([
            self.pool(tensor),
            self.out1(inputs),
            self.out6(inputs),
            self.out12(
                inputs
            ),
            self.out18(
                inputs
            )
        ])
        tensor = self.conv2(tensor)
        return tensor

@register_keras_serializable()
class DeeplabV3Plus(tf.keras.Model):
    def __init__(self, num_classes=4, **kwargs):
        super(DeeplabV3Plus, self).__init__()

        self.num_classes = num_classes
        self.aspp = None
        self.backbone_feature_1, self.backbone_feature_2 = None, None
        self.input_a_upsampler_getter = None
        self.otensor_upsampler_getter = None
        self.input_b_conv, self.conv1, self.conv2, self.out_conv = (None,
                                                                    None,
                                                                    None,
                                                                    None)

    @staticmethod
    def _get_conv_block(filters, kernel_size, conv_activation=None):
        return ConvBlock(filters, kernel_size=kernel_size, padding='same',
                         conv_activation=conv_activation,
                         kernel_initializer=tf.keras.initializers.he_normal(),
                         use_bias=False, dilation_rate=1)

    @staticmethod
    def _get_upsample_layer_fn(input_shape, factor: int):
        return lambda fan_in_shape: \
            tf.keras.layers.UpSampling2D(
                size=(
                    input_shape[1] // factor // fan_in_shape[1],
                    input_shape[2] // factor // fan_in_shape[2]
                ),
                interpolation='bilinear'
            )

    def _get_backbone_feature(self, input_shape, feature_name) -> tf.keras.Model:
        input_layer = tf.keras.Input(shape=input_shape[1:])
        backbone = ResNet50(input_tensor=input_layer, weights='imagenet', include_top=False)
        output_layer = backbone.get_layer(feature_name).output
        return tf.keras.Model(inputs=input_layer, outputs=output_layer)

    def build(self, input_shape):
        self.backbone_feature_1 = self._get_backbone_feature(input_shape, 'conv4_block6_out')
        self.backbone_feature_2 = self._get_backbone_feature(input_shape, 'conv2_block3_out')

        self.input_a_upsampler_getter = self._get_upsample_layer_fn(input_shape, factor=4)
        self.aspp = AtrousSpatialPyramidPooling()
        self.input_b_conv = DeeplabV3Plus._get_conv_block(48, kernel_size=(1, 1))
        self.conv1 = DeeplabV3Plus._get_conv_block(256, kernel_size=3, conv_activation='relu')
        self.conv2 = DeeplabV3Plus._get_conv_block(256, kernel_size=3, conv_activation='relu')

        self.otensor_upsampler_getter = self._get_upsample_layer_fn(input_shape, factor=1)
        self.out_conv = tf.keras.layers.Conv2D(self.num_classes, kernel_size=(1, 1), padding='same', activation='softmax', dtype='float32')

    def call(self, inputs, training=False, mask=None):
        input_a = self.backbone_feature_1(inputs, training=training)
        input_a = self.aspp(input_a)

        input_b = self.backbone_feature_2(inputs, training=training)
        input_b = self.input_b_conv(input_b)

        input_b = tf.image.resize(input_b, size=tf.shape(input_a)[1:3], method='bilinear')

        tensor = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])

        tensor = self.conv1(tensor)
        tensor = self.conv2(tensor)

        output = self.out_conv(tensor)

        output = tf.image.resize(output, size=tf.shape(inputs)[1:3], method='bilinear')

        return output