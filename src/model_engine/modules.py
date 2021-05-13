# -----------------------------------------------------------------------------
# This software was developed as practical part of Master's thesis at FIT BUT
# The program uses multiobjective NSGA-II algorithm for designing accurate
# and compact CNNs.
#
# Author: Jan Pristas, xprist06@stud.fit.vutbr.cz
# Institute: Faculty of Information Technology, Brno University of Technology
#
# File: modules.py
# Description: Add phase modules into CNN structure
# -----------------------------------------------------------------------------


from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate


def conv_module(x, k, kernel_size, strides=(1, 1), activation="relu", kernel_initializer="he_uniform", padding="same"):
    """
    Add convolutional module into phase CNN structure
    :param x: Model structure
    :param k: Number of output filters in the convolution
    :param kernel_size: Tuple of 2 integers, specifying the height and width of the 2D convolution window
    :param strides: Tuple of 2 integers, specifying the strides of the convolution along the height and width, default value is (1, 1)
    :param activation: Activation function, default value is "relu"
    :param kernel_initializer: Kernel initializer, default value if "he_uniform"
    :param padding: Padding, default value is "same"
    :return: Phase CNN structure with convolutional module added
    """
    x = Conv2D(k, kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(0.1)(x)
    return x


def inception_module(x, k_1x1, k_3x3, k_5x5):
    """
    Add inception module into phase CNN structure
    :param x: Model structure
    :param k_1x1: Number of output filters for 1x1 convolution
    :param k_3x3: Number of output filters for 3x3 convolution
    :param k_5x5: Number of putput filters for 5x5 convolution
    :return: Phase CNN structure with inception module added
    """
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
    """
    Add fully connected module at the end of CNN structure
    :param x: Model structure
    :param classes_cnt: Number of classes for last layer
    :return: Complete CNN structure
    """
    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.25)(x)
    x = Dense(classes_cnt, activation='softmax')(x)
    return x
