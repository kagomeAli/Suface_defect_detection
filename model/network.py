import numpy as np
import os, gc, cv2, shutil,platform

import tensorflow as tf
import keras.backend as K
import tensorflow_addons as tfa
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from skimage.feature import local_binary_pattern

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose,Dropout, concatenate, BatchNormalization, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
#import seaborn as sns
import pandas as pd


from tensorflow_addons.layers import InstanceNormalization, GroupNormalization, SpatialPyramidPooling2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from sklearn.model_selection import train_test_split

import keras, gzip, pickle
from keras.layers import *
from keras.optimizers import *
from keras.models import Model, load_model
from keras.callbacks import *
from sklearn.preprocessing import OneHotEncoder


from sklearn.utils import shuffle


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


import util as ul
from .LBC import LBC
from .attention_module import attach_attention_module

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D


##  R2AttUNet start
#coding=utf-8
import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *

def conv_block(input, filters):
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def up_conv(input, filters):
    out = UpSampling2D()(input)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def Attention_block(input1, input2, filters):
    g1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input1)
    g1 = BatchNormalization()(g1)
    x1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input2)
    x1 = BatchNormalization()(x1)
    psi = Activation('relu')(add([g1, x1]))
    psi = Conv2D(filters, kernel_size=1, strides=1, padding='same')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    out = multiply([input2, psi])
    return out

def Recurrent_block(input, channel, t=2):
    for i in range(t):
        if i == 0:
            x = Conv2D(channel, kernel_size=(3, 3), strides=1, padding='same')(input)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        out = Conv2D(channel, kernel_size=(3, 3), strides=1, padding='same')(add([x, x]))
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
    return out

def RRCNN_block(input, channel, t=2):
    x1 = Conv2D(channel, kernel_size=(1, 1), strides=1, padding='same')(input)
    x2 = Recurrent_block(x1, channel, t=t)
    x2 = Recurrent_block(x2, channel, t=t)
    out = add([x1, x2])
    return out



def R2AttUNet(nClasses, input_shape = (256, 256, 1)):
    # """
    #Residual Recuurent Block with attention Unet
    #Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    #"""
    inputs = Input(shape=input_shape)
    t = 2
    n1 = 32
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

    e1 = RRCNN_block(inputs, filters[0], t=t)

    e2 = MaxPooling2D(strides=2)(e1)
    e2 = RRCNN_block(e2, filters[1], t=t)

    e3 = MaxPooling2D(strides=2)(e2)
    e3 = RRCNN_block(e3, filters[2], t=t)

    e4 = MaxPooling2D(strides=2)(e3)
    e4 = RRCNN_block(e4, filters[3], t=t)

    e5 = MaxPooling2D(strides=2)(e4)
    e5 = RRCNN_block(e5, filters[4], t=t)

    d5 = up_conv(e5, filters[3])
    x4 =  Attention_block(d5, e4, filters[3])
    d5 = Concatenate()([x4, d5])
    d5 = conv_block(d5, filters[3])

    d4 = up_conv(d5, filters[2])
    x3 =  Attention_block(d4, e3, filters[2])
    d4 = Concatenate()([x3, d4])
    d4 = conv_block(d4, filters[2])

    d3 = up_conv(d4, filters[1])
    x2 =  Attention_block(d3, e2, filters[1])
    d3 = Concatenate()([x2, d3])
    d3 = conv_block(d3, filters[1])

    d2 = up_conv(d3, filters[0])
    x1 =  Attention_block(d2, e1, filters[0])
    d2 = Concatenate()([x1, d2])
    d2 = conv_block(d2, filters[0])

    out = Conv2D(nClasses, (3, 3), padding='same')(d2)


    model = Model(inputs=inputs, outputs=out)

    return model
##  R2AttUNet end

## Deeplab v3 start

from tensorflow.python.keras.utils import conv_utils
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.utils import conv_utils

from keras.utils.data_utils import get_file
from keras.utils.conv_utils import normalize_data_format


class BilinearUpsampling(Layer):

    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
                 input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0], height, width, input_shape[3])

    def call(self, inputs):
        # .tf
        return tf.compat.v1.image.resize_bilinear(inputs, (int(inputs.shape[1] * self.upsampling[0]),
                                                 int(inputs.shape[2] * self.upsampling[1])))

    def get_config(self):

        config = {'size': self.upsampling, 'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def xception_downsample_block(x, channels, top_relu=False):
    ##separable conv1
    if top_relu:
        x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    ##separable conv2
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    ##separable conv3
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def res_xception_downsample_block(x, channels):
    res = Conv2D(channels, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = xception_downsample_block(x, channels)
    x = add([x, res])
    return x


def xception_block(x, channels):
    ##separable conv1
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    ##separable conv2
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    ##separable conv3
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def res_xception_block(x, channels):
    res = x
    x = xception_block(x, channels)
    x = add([x, res])
    return x


def aspp(x, input_shape, out_stride = 8):
    b0 = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    b0 = BatchNormalization()(b0)
    b0 = Activation("relu")(b0)

    b1 = DepthwiseConv2D((3, 3), dilation_rate=(6, 6), padding="same", use_bias=False)(x)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)
    b1 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)

    b2 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)
    b2 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)

    b3 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)
    b3 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)

    out_shape = int(input_shape[0] / out_stride)
    b4 = AveragePooling2D(pool_size=(out_shape, out_shape))(x)
    b4 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b4)
    b4 = BatchNormalization()(b4)
    b4 = Activation("relu")(b4)
    b4 = BilinearUpsampling((out_shape, out_shape))(b4)

    x = Concatenate()([b4, b0, b1, b2, b3])
    return x


def DeeplabV3_plus(nClasses=21, input_shape = (256, 256, 1), out_stride=16):
    img_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding="same", use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = res_xception_downsample_block(x, 128)

    res = Conv2D(256, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    skip = BatchNormalization()(x)
    x = Activation("relu")(skip)
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = add([x, res])

    x = xception_downsample_block(x, 728, top_relu=True)

    for i in range(16):
        x = res_xception_block(x, 728)

    res = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    res = BatchNormalization()(res)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(728, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = add([x, res])

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1536, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1536, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(2048, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # aspp
    x = aspp(x, input_shape, out_stride)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.9)(x)

    ##decoder
    x = BilinearUpsampling((4, 4))(x)
    dec_skip = Conv2D(48, (1, 1), padding="same", use_bias=False)(skip)
    dec_skip = BatchNormalization()(dec_skip)
    dec_skip = Activation("relu")(dec_skip)
    x = Concatenate()([x, dec_skip])

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(nClasses, (1, 1), padding="same")(x)
    x = BilinearUpsampling((4, 4))(x)
#     outputHeight = Model(img_input, x).output_shape[1]
#     outputWidth = Model(img_input, x).output_shape[2]
#     x = (Reshape((outputHeight * outputWidth, nClasses)))(x)
#     x = Activation('softmax')(x)
    model = Model(img_input, x)
    return model
## deeplab end


## Unet


from tensorflow.keras import layers

def Unet(input_shape = (256,256, 1), num_classes = 2):
    inputs = keras.Input(shape=input_shape)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

## LBCNN3

def residual_dense_block(input_layer, start_neurons = 64):

    x1_1 = Conv2D(start_neurons, (3, 3), activation="relu", padding="same")(input_layer)

    x1_1_concat = concatenate([input_layer, x1_1])
    x1_2 = Conv2D(start_neurons, (3, 3), activation="relu", padding="same")(x1_1_concat)

    x1_2_concat = concatenate([input_layer, x1_1, x1_2])
    x1_3 = Conv2D(start_neurons, (3, 3), activation="relu", padding="same")(x1_2_concat)

    x1_3_concat = concatenate([input_layer, x1_1, x1_2, x1_3])
    x1_3 = Conv2D(start_neurons, (1, 1), activation="relu", padding="same")(x1_3_concat)

    output_layer = concatenate([input_layer, x1_3])

    return output_layer

def LBCNN3(input_shape=(256, 256, 1),
           kernel=3,
           start_neurons=64,
           classes=10):
    img_input = Input(shape=input_shape, name='input')
    # attach_attention_module
    # block 1
    conv1_first = residual_dense_block(img_input)
    conv1 = BatchNormalization()(conv1_first)
    conv1 = attach_attention_module(conv1)
    conv1 = Conv2D(start_neurons * 1, (kernel, kernel), activation="relu", padding="same")(conv1)
    #     conv1 = Conv2D(start_neurons * 1, (kernel, kernel), activation="relu", padding="same")(conv1)

    resodual_layer1 = concatenate([conv1_first, conv1])
    conv1_last = Activation('relu')(resodual_layer1)
    pool1 = MaxPooling2D((2, 2))(conv1_last)

    # block 2
    conv2_first = residual_dense_block(pool1)
    conv2 = BatchNormalization()(conv2_first)
    conv2 = attach_attention_module(conv2)
    conv2 = Conv2D(start_neurons * 2, (kernel, kernel), activation="relu", padding="same")(conv2)
    #     conv2 = Conv2D(start_neurons * 2, (kernel, kernel), activation="relu", padding="same")(conv2)

    resodual_layer2 = concatenate([conv2_first, conv2])
    conv2_last = Activation('relu')(resodual_layer2)
    pool2 = MaxPooling2D((2, 2))(conv2_last)

    # block 3
    conv3_first = residual_dense_block(pool2)
    conv3 = BatchNormalization()(conv3_first)
    conv3 = attach_attention_module(conv3)
    conv3 = Conv2D(start_neurons * 4, (kernel, kernel), activation="relu", padding="same")(conv3)
    #     conv3 = Conv2D(start_neurons * 4, (kernel, kernel), activation="relu", padding="same")(conv3)

    resodual_layer3 = concatenate([conv3_first, conv3])
    conv3_last = Activation('relu')(resodual_layer3)
    pool3 = MaxPooling2D((2, 2))(conv3_last)

    # middle
    convm = Conv2D(start_neurons * 8, (kernel, kernel), activation="relu", padding="same")(pool3)
    convm = Conv2D(start_neurons * 8, (kernel, kernel), activation="relu", padding="same")(convm)
    deconv3 = Conv2DTranspose(start_neurons * 4, (kernel, kernel), strides=(2, 2), padding="same")(convm)

    # unblock 3
    uconv3_resodual = concatenate([conv3_first, deconv3])
    uconv3_resodual = attach_attention_module(uconv3_resodual)
    uconv3 = Conv2D(start_neurons * 4, (kernel, kernel), activation="relu", padding="same")(uconv3_resodual)
    #     uconv3 = Conv2D(start_neurons * 4, (kernel, kernel), activation="relu", padding="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    deconv2 = Conv2DTranspose(start_neurons * 2, (kernel, kernel), strides=(2, 2), padding="same")(uconv3)

    # unblock 2
    uconv2_resodual = concatenate([conv2_first, deconv2])
    uconv2_resodual = attach_attention_module(uconv2_resodual)
    uconv2 = Conv2D(start_neurons * 2, (kernel, kernel), activation="relu", padding="same")(uconv2_resodual)
    #     uconv2 = Conv2D(start_neurons * 2, (kernel, kernel), activation="relu", padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (kernel, kernel), strides=(2, 2), padding="same")(uconv2)

    # unblock 1
    uconv1_resodual = concatenate([conv1_first, deconv1])
    uconv1_resodual = attach_attention_module(uconv1_resodual)
    uconv1 = Conv2D(start_neurons * 1, (kernel, kernel), activation="relu", padding="same")(uconv1_resodual)
    #     uconv1 = Conv2D(start_neurons * 1, (kernel, kernel), activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)

    # output
    output_layer = Conv2D(2, (1, 1), padding="same", activation="sigmoid", name='output')(uconv1)

    model = Model(img_input, output_layer, name='CNN')
    return model


##
def LBCNN2(input_shape = (256, 256, 1),
          kernel = 3,
          start_neurons = 64,
          classes=2):


    img_input = Input(shape=input_shape, name = 'input')
# attach_attention_module
    # block 1
    conv1_first = residual_dense_block(img_input)
    conv1 = BatchNormalization()(conv1_first)
    conv1 = attach_attention_module(conv1)
    conv1 = Conv2D(start_neurons * 1, (kernel, kernel), activation="relu", padding="same")(conv1)
#     conv1 = Conv2D(start_neurons * 1, (kernel, kernel), activation="relu", padding="same")(conv1)

    resodual_layer1 = concatenate([conv1_first, conv1])
    conv1_last = Activation('relu')(resodual_layer1)
    pool1 = MaxPooling2D((2, 2))(conv1_last)

    # block 2
    conv2_first = residual_dense_block(pool1)
    conv2 = BatchNormalization()(conv2_first)
    conv2 = attach_attention_module(conv2)
    conv2 = Conv2D(start_neurons * 2, (kernel, kernel), activation="relu", padding="same")(conv2)
#     conv2 = Conv2D(start_neurons * 2, (kernel, kernel), activation="relu", padding="same")(conv2)


    resodual_layer2 = concatenate([conv2_first, conv2])
    conv2_last = Activation('relu')(resodual_layer2)
    pool2 = MaxPooling2D((2, 2))(conv2_last)

    # block 3
    conv3_first = residual_dense_block(pool2)
    conv3 = BatchNormalization()(conv3_first)
    conv3 = attach_attention_module(conv3)
    conv3 = Conv2D(start_neurons * 4, (kernel, kernel), activation="relu", padding="same")(conv3)
#     conv3 = Conv2D(start_neurons * 4, (kernel, kernel), activation="relu", padding="same")(conv3)

    resodual_layer3 = concatenate([conv3_first, conv3])
    conv3_last = Activation('relu')(resodual_layer3)
    pool3 = MaxPooling2D((2, 2))(conv3_last)

    # middle
    convm = Conv2D(start_neurons * 8, (kernel, kernel), activation="relu", padding="same")(pool3)
    convm = Conv2D(start_neurons * 8, (kernel, kernel), activation="relu", padding="same")(convm)
    deconv3 = Conv2DTranspose(start_neurons * 4, (kernel, kernel), strides=(2, 2), padding="same")(convm)

    # unblock 3
    uconv3_resodual = concatenate([conv3_first, deconv3])
    uconv3_resodual = attach_attention_module(uconv3_resodual)
    uconv3 = Conv2D(start_neurons * 4, (kernel, kernel), activation="relu", padding="same")(uconv3_resodual)
#     uconv3 = Conv2D(start_neurons * 4, (kernel, kernel), activation="relu", padding="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    deconv2 = Conv2DTranspose(start_neurons * 2, (kernel, kernel), strides=(2, 2), padding="same")(uconv3)

    # unblock 2
    uconv2_resodual = concatenate([conv2_first, deconv2])
    uconv2_resodual = attach_attention_module(uconv2_resodual)
    uconv2 = Conv2D(start_neurons * 2, (kernel, kernel), activation="relu", padding="same")(uconv2_resodual)
#     uconv2 = Conv2D(start_neurons * 2, (kernel, kernel), activation="relu", padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (kernel, kernel), strides=(2, 2), padding="same")(uconv2)

    # unblock 1
    uconv1_resodual = concatenate([conv1_first, deconv1])
    uconv1_resodual = attach_attention_module(uconv1_resodual)
    uconv1 = Conv2D(start_neurons * 1, (kernel, kernel), activation="relu", padding="same")(uconv1_resodual)
#     uconv1 = Conv2D(start_neurons * 1, (kernel, kernel), activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)

    # output
    output_layer = Conv2D(classes, (1, 1), padding="same", activation="sigmoid",  name = 'output')(uconv1)


    model = Model(img_input, output_layer, name='CNN')
    return model

## Mobilenet v3

from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D,UpSampling2D, Concatenate
from keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape
from keras import backend as K
from keras.models import Model
import tensorflow as tf
# 定义relu6激活函数
def relu6(x):
    return K.relu(x, max_value=6.0)
# 定义h-swish激活函数
def hard_swish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0
# 定义返回的激活函数是relu6还是h-swish
def return_activation(x, nl):
    if nl == 'HS':
        x = Activation(hard_swish)(x)
    if nl == 'RE':
        x = Activation(relu6)(x)
    return x
# 定义卷积块(卷积+标准化+激活函数)
def conv_block(inputs, filters, kernel, strides, nl):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return return_activation(x, nl)
# 定义注意力机制模块
def SE(inputs):
    input_channels = int(inputs.shape[-1])
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(input_channels, activation='relu')(x)
    x = Dense(input_channels, activation='hard_sigmoid')(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])
    return x

def bottleneck(inputs, filters, kernel, e, s, squeeze, nl,alpha=1.0):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    input_shape = K.int_shape(inputs)
    tchannel = int(e)
    cchannel = int(alpha * filters)
    r = s == 1 and input_shape[3] == filters
    x = conv_block(inputs, tchannel, (1,1), (1,1), nl)
    x = DepthwiseConv2D(kernel, strides=(s,s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = [](x,nl)
    if squeeze:
        x = SE(x)
    x = Conv2D(cchannel, (1,1), strides=(1,1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    if r:
        x = Add()([x, inputs])
    return x

def MobileNetV3_Large(inputs, alpha=1.0):
    # conv2d_1 (Conv2D) - activation_1 (Activation)
    x = conv_block(inputs, 16, (3,3), strides=(2,2), nl='HS')
    # conv2d_2 (Conv2D) - add_1 (Add)
    x = bottleneck(x, 16, (3, 3), e=16, s=1, squeeze=False, nl='RE', alpha=alpha)
    # conv2d_4 (Conv2D) - batch_normalization_7
    x = bottleneck(x, 24, (3, 3), e=64, s=2, squeeze=False, nl='RE', alpha=alpha)
    # conv2d_6 (Conv2D) - add_2 (Add)
    x = bottleneck(x, 24, (3, 3), e=72, s=1, squeeze=False, nl='RE', alpha=alpha)
    # conv2d_8 (Conv2D) - batch_normalization_13
    x = bottleneck(x, 40, (5, 5), e=72, s=2, squeeze=True, nl='RE', alpha=alpha)
    # conv2d_10 (Conv2D) - add_3 (Add)
    x = bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE', alpha=alpha)
    # conv2d_12 (Conv2D) - add_4 (Add)   (None, 52, 52, 40)    70层
    x = bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE', alpha=alpha)



    # conv2d_14 (Conv2D) - batch_normalization_22  (None, 26, 26, 80)
    x = bottleneck(x, 80, (3, 3), e=240, s=2, squeeze=False, nl='HS', alpha=alpha)
    # conv2d_16 (Conv2D) - add_5 (Add)   (None, 26, 26, 80)
    x = bottleneck(x, 80, (3, 3), e=200, s=1, squeeze=False, nl='HS', alpha=alpha)
    # conv2d_18 (Conv2D) - add_6 (Add)
    x = bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS', alpha=alpha)
    # conv2d_20 (Conv2D) - add_7 (Add)    (None, 26, 26, 80)
    x = bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS', alpha=alpha)
    # conv2d_22 (Conv2D) - batch_normalization_34   (None, 26, 26, 112)
    # inputs=(26,26,80) filtters=112 所以没有Add
    x = bottleneck(x, 112, (3, 3), e=480, s=1, squeeze=True, nl='HS', alpha=alpha)
    # conv2d_24 (Conv2D) - add_8 (Add)   (None, 26, 26, 112)     132层
    x = bottleneck(x, 112, (3, 3), e=672, s=1, squeeze=True, nl='HS', alpha=alpha)



    # conv2d_26 (Conv2D) - batch_normalization_40   (None, 13, 13, 160)
    x = bottleneck(x, 160, (5, 5), e=672, s=2, squeeze=True, nl='HS', alpha=alpha)
    # conv2d_28 (Conv2D) - add_9 (Add)   (None, 13, 13, 160)
    x = bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS', alpha=alpha)
    # conv2d_30 (Conv2D) - add_10 (Add)  (None, 13, 13, 160)
    x = bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS', alpha=alpha)
    model = Model(inputs, x)
    return model

def MobileNetV3_Small(inputs, alpha=1.0):
    x = conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')
    x = bottleneck(x, 16, (3, 3), e=16, s=2, squeeze=True, nl='RE', alpha=alpha)
    x = bottleneck(x, 24, (3, 3), e=72, s=2, squeeze=False, nl='RE', alpha=alpha)
    x = bottleneck(x, 24, (3, 3), e=88, s=1, squeeze=False, nl='RE', alpha=alpha)
    x = bottleneck(x, 40, (5, 5), e=96, s=2, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 48, (5, 5), e=120, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 48, (5, 5), e=144, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 96, (5, 5), e=288, s=2, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS', alpha=alpha)
    model = Model(inputs, x)
    return model

def make_last_layers(x, num_filters, out_filters):
    x = conv_block(x, num_filters, (1, 1), strides=(1, 1), nl='HS')
    x = conv_block(x, num_filters, (3, 3), strides=(1, 1), nl='HS')
    x = conv_block(x, num_filters, (1, 1), strides=(1, 1), nl='HS')
    y = conv_block(x, num_filters, (3, 3), strides=(1, 1), nl='HS')
    y = Conv2D(out_filters, (1, 1), strides=(1, 1), padding='same')(y)
    return x, y

def yolo_body(inputs, num_anchors, num_classes):
    mobilenet = MobileNetV3_Large(inputs)
    x, y1 = make_last_layers(mobilenet.output, 160, num_anchors*(num_classes+5))
    x = conv_block(x, 112, (1, 1), strides=(1, 1),nl='HS')
    x = UpSampling2D(2)(x)
    x = Concatenate()([x,mobilenet.layers[132].output])
    x, y2 = make_last_layers(x, 112, num_anchors*(num_classes+5))
    x = conv_block(x, 80, (1, 1), strides=(1, 1),nl='HS')
    x = UpSampling2D(2)(x)
    x = Concatenate()([x,mobilenet.layers[70].output])
    x, y3 = make_last_layers(x, 80, num_anchors*(num_classes+5))
    return Model(inputs, [y1,y2,y3])

def tiny_yolo_body(inputs, num_anchors, num_classes):
    mobilenet = MobileNetV3_Small(inputs)
    x = conv_block(mobilenet.output, 96, (3, 3), strides=(1, 1), nl='HS')
    y1 = Conv2D(num_anchors*(num_classes+5), (1, 1), strides=(1, 1), padding='same')(x)
    x = UpSampling2D(2)(mobilenet.output)
    x = Concatenate()([x, mobilenet.layers[104].output])
    x = conv_block(x, 96, (3, 3), strides=(1, 1), nl='HS')
    y2 = Conv2D(num_anchors*(num_classes+5), (1, 1), strides=(1, 1), padding='same')(x)
    return Model(inputs, [y1,y2])

def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))
    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=50,
              score_threshold=.6,
              iou_threshold=.4):
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)
    return boxes_, scores_, classes_