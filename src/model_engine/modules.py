from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate


def conv_module(x, k, kernel_size, strides=(1, 1), activation="relu", kernel_initializer="he_uniform", padding="same"):
    x = Conv2D(k, kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(0.1)(x)
    return x


def inception_module(x, k_1x1, k_3x3, k_5x5):
    conv_1x1 = conv_module(x, k_1x1, (1, 1))
    # CONV 1x1 => CONV 3x3
    conv_1x1_1 = conv_module(x, k_1x1, (1, 1))
    conv_3x3 = conv_module(conv_1x1_1, k_3x3, (3, 3))
    # CONV 1x1 => CONV 5x5
    conv_1x1_2 = conv_module(x, k_1x1, (1, 1))
    conv_5x5 = conv_module(conv_1x1_2, k_5x5, (5, 5))
    # MAX_POOL 3x3 => CONV 1x1
    max_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    conv_1x1_3 = conv_module(max_pool, k_1x1, (1, 1))
    x = concatenate([conv_1x1, conv_3x3, conv_5x5, conv_1x1_3])
    return x


def fully_connected_module(x, classes_cnt):
    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.25)(x)
    x = Dense(classes_cnt, activation='softmax')(x)
    return x
