#####################
# 1. Library Import #
#####################
import numpy as np
import os, matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import regularizers
from tensorflow.keras import utils


def conv_block_2d(lr_conv, lr_num, par_list):
    # parameter
    conv_size = par_list[0]
    conv_str = par_list[1]
    conv_act = par_list[2]
    pool_str = par_list[3]
    reg_weight = par_list[4]
    # code
    for i in range(lr_num):
        lr_conv = layers.Conv2D(conv_size, conv_str, activation=None, padding='same',
                                kernel_regularizer=reg_weight, kernel_initializer='he_normal')(lr_conv)
        lr_conv = layers.BatchNormalization(axis=-1)(lr_conv)
        lr_conv = layers.Activation(conv_act)(lr_conv)
    lr_pool = layers.MaxPooling2D(pool_size=pool_str)(lr_conv)
    return lr_pool


def output_block(lr_dense, block_num, flat_count, reg_weight, act_func, drop_rate):
    lr_dense = layers.Flatten()(lr_dense)
    lr_dense = layers.Dropout(drop_rate)(lr_dense)
    for i in range(block_num):
        lr_dense = layers.Dense(flat_count[i], kernel_regularizer=reg_weight,
                                activation=None)(lr_dense)
        lr_dense = layers.BatchNormalization(axis=-1)(lr_dense)
        lr_dense = layers.Activation(act_func)(lr_dense)
        lr_dense = layers.Dropout(drop_rate)(lr_dense)
    return lr_dense

################
# 2. Data Load #
################
# tf.debugging.set_log_device_placement(True)
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    (x_train, y_train), (x_test, y_test)=keras.datasets.mnist.load_data(path='minist.npz')
    print(x_train.shape, y_train.shape)

    x_train_list = []
    x_test_list = []
    for i, i_ in enumerate(x_train[:1000]):
        arr = np.zeros(shape=(32, 32))
        arr[:28,:28] = x_train[i]
        x_train_list.append(arr)
    for i, i_ in enumerate(x_test[:500]):
        arr = np.zeros(shape=(32, 32))
        arr[:28,:28] = x_test[i]
        x_test_list.append(arr)

    x_train1 = np.expand_dims(np.array(x_train_list), axis=-1)
    x_test1 = np.expand_dims(np.array(x_test_list), axis=-1)
    print('Image Shape = ' ,x_train1.shape, x_test1.shape)

    y_train_list = []
    y_test_list = []
    for i, i_ in enumerate(y_train[:1000]):
        zero = [0] * 10
        zero[i_] = 1
        y_train_list.append(zero)

    for i, i_ in enumerate(y_test[:500]):
        zero = [0] * 10
        zero[i_] = 1
        y_test_list.append(zero)

    y_train1 = np.array(y_train_list)
    y_test1 = np.array(y_test_list)
    print('Label Shape', y_train1.shape, y_test1.shape)

    #################
    # Network Build #
    #################
    inputs = Input(shape=(32,32,1))
    block1 = conv_block_2d(inputs, 2, [64, 3, 'relu', 2, None])
    block2 = conv_block_2d(block1, 2, [128, 3, 'relu', 2, None])
    block3 = conv_block_2d(block2, 3, [256, 3, 'relu', 2, None])
    block4 = conv_block_2d(block3, 3, [512, 3, 'relu', 2, None])
    block5 = conv_block_2d(block4, 3, [512, 3, 'relu', 2, None])
    dens = output_block(block5, 1, [256], None, 'relu', 0.5)
    outputs = layers.Dense(10, activation='softmax')(dens)
    model = Model(inputs, outputs)
    model.summary()

    ###############
    # Train Model #
    ###############
    model.compile(loss=losses.CategoricalCrossentropy(), optimizer=optimizers.Adam(lr=1e-4), metrics=['accuracy'])
    history = model.fit(x_train1, y_train1, epochs=20, batch_size=32, validation_data=(x_test1, y_test1), shuffle=True)