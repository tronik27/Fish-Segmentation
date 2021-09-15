from typing import Tuple
from tensorflow.keras.layers import BatchNormalization, Conv2D, LeakyReLU, Input,\
    concatenate, UpSampling2D, AveragePooling2D, Dense, GlobalAveragePooling2D, Conv2DTranspose
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
from tensorflow.keras.applications import DenseNet121


class DenseDeepLabV3Plus:
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, num_filters: int,
                 input_name: str, output_name: str):
        """
        Custom implementation of the Deeplab V3+, using DenseNet121 model as backbone and adding extra output for image
        classification for small image segmentation and classification task.
        :param input_shape: input shape (height, width, channels).
        :param num_classes: number of classes.
        :param num_filters: network expansion factor, determines the number of filters in custom model layers.
        :param input_name: name of the input tensor.
        :param output_name: name of the output tensor.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.input_name = input_name
        self.output_name = output_name
        self.num_filters = num_filters
        self.conv_kwargs = {'use_bias': False, 'padding': 'same', 'kernel_initializer': 'he_normal'}

    def build(self) -> tf.keras.models.Model:
        """
        Building CNN model.
        :return: Model() object.
        """
        inputs = Input(shape=self.input_shape, name=self.input_name)

        encoder = DenseNet121(include_top=False, weights="imagenet", input_tensor=inputs)
        layers_names = [layer.name for layer in encoder.layers]
        for layer in encoder.layers[:layers_names.index('pool4_relu') + 1]:
            layer.trainable = False

        x1 = encoder.get_layer('pool4_relu').output
        x1 = self.aspp_block(x=x1, filters=self.num_filters)
        x1 = UpSampling2D(size=(4, 4), interpolation='bilinear')(x1)

        x2 = encoder.get_layer('pool2_relu').output
        x2 = self.conv_block(x=x2, filters=48, kernel_size=1)
        x = concatenate([x1, x2])

        x = self.conv_block(x=x, filters=self.num_filters, kernel_size=3, apply_conv_activation=True)
        x = self.conv_block(x=x, filters=self.num_filters, kernel_size=3, apply_conv_activation=True)
        x = self.conv_block(x=x, filters=self.num_filters, kernel_size=3, apply_conv_activation=True)
        x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

        mask_output = Conv2D(filters=1, kernel_size=(1, 1), strides=1, activation='sigmoid', name='masks')(x)
        label_output = self.classification_output(base_model=encoder)

        return Model(inputs=inputs, outputs=[mask_output, label_output])

    def conv_block(self, x: Tensor, filters: int, kernel_size: int, dilation_rate: int = 1,
                   apply_conv_activation: bool = False) -> Tensor:
        """
        Convolution block.
        :param x: input tensor.
        :param filters: number of filters in output tensor.
        :param kernel_size: number of filters in output tensor.
        :param dilation_rate: convolution dilation rate.
        :param apply_conv_activation: indicates whether apply activation after convolution layer or not.
        :return: output tensor.
        """
        x = Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, **self.conv_kwargs)(x)
        if apply_conv_activation:
            x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    def aspp_block(self, x: Tensor, filters: int) -> Tensor:
        """
        Atrous Spatial Pyramid Pooling block.
        :param x: input tensor.
        :param filters: number of filters in output tensor.
        :return: output tensor.
        """
        kernels = [1, 3, 3, 3]
        dilation_rates = [1, 6, 12, 18]

        pool = AveragePooling2D(pool_size=(x.shape[1], x.shape[2]))(x)
        pool = self.conv_block(x=pool, filters=filters, kernel_size=1)
        pool = UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation='bilinear')(pool)
        assp = [pool]

        for kernel, dilation_rate in zip(kernels, dilation_rates):
            assp.append(self.conv_block(x=pool, filters=filters, kernel_size=kernel, dilation_rate=dilation_rate))

        x = concatenate(assp)
        x_out = self.conv_block(x=x, filters=filters, kernel_size=1)
        return x_out

    def classification_output(self, base_model: tf.keras.Model) -> Tensor:
        """
        Creating output for image classification.
        :param base_model: backbone model.
        :return: output tensor with labels predictions.
        """
        x = base_model.get_layer('relu').output
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=self.num_classes, activation='softmax', name='labels')(x)
        return x
