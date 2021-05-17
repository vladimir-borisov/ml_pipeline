from tensorflow.keras.layers import (Input, Conv2D, MaxPool2D, ReLU, Dense, Flatten)
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf

import yaml
from addict import Dict

# read config file
with open('./configs/train_params.yaml') as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

config = Dict(config)


def get_model() -> tf.keras.Model:

    inputs = Input(shape=(config.image_width, config.image_height, config.image_channels))

    block1 = Sequential(layers = [Conv2D(16, 7, activation='relu', padding='same', kernel_initializer='he_normal'),
                                  MaxPool2D(pool_size=(2, 2)),
                                  ReLU()])

    block2 = Sequential(layers = [Conv2D(8, 5, activation='relu', padding='same', kernel_initializer='he_normal'),
                                  MaxPool2D(pool_size=(2, 2)),
                                  ReLU()])

    output = block1(inputs)
    output = block2(output)
    output = Flatten()(output)
    output = Dense(units=2, activation='softmax')(output)


    return tf.keras.Model(inputs=inputs, outputs=output)
