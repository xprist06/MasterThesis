from model_engine import modules, layers
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, concatenate

import tensorflow as tf


class Phase:
    def __init__(self, phase_connections):
        self.phase_connections = phase_connections
        self.output_module_idx = len(phase_connections) - 2
        self.phase_skip_idx = len(phase_connections) - 1
        self.modules = []

    def is_empty(self):
        is_empty = True
        for i in range(len(self.phase_connections)):
            if i == self.output_module_idx:
                continue
            if len(self.phase_connections[i]) > 1 or self.phase_connections[i] != [0]:
                is_empty = False
                break
        return is_empty

    def create_phase_model(self, x):
        phase_input = x
        phase_output = None
        for i in range(len(self.phase_connections) - 1):
            if i != self.output_module_idx and self.phase_connections[i][0] == 0:
                self.modules.append(None)
            elif i == 0:
                x = self.add_first_module(x)
            elif i == self.output_module_idx:
                phase_output = self.add_last_module(x, phase_input)
            else:
                x = self.add_module(x, phase_input, i)

        if phase_output is None:
            return phase_input

        return phase_output

    def add_first_module(self, x):
        module = layers.phase_layers[0]
        return self.choose_layer(x, x, module)

    def add_last_module(self, x, phase_input):
        inputs = []
        for i in range(self.output_module_idx):
            if self.phase_connections[self.output_module_idx][i] == 1:
                inputs.append(self.modules[i])
        if self.phase_connections[self.phase_skip_idx][0] == 1:
            inputs.append(phase_input)
        if len(inputs) > 1:
            x = concatenate(inputs)
        else:
            x = tf.identity(x)
        return x

    def add_module(self, x, phase_input, layer_index):
        module = layers.phase_layers[layer_index]
        inputs = []
        for i in range(1, len(self.phase_connections[layer_index])):
            if self.phase_connections[layer_index][i] == 1:
                inputs.append(self.modules[i - 1])
        if len(inputs) == 0:
            return self.choose_layer(x, phase_input, module)
        elif len(inputs) == 1:
            x = inputs[0]
        elif len(inputs) > 1:
            x = concatenate(inputs)
        return self.choose_layer(x, x, module)

    def choose_layer(self, x, layer_input, module):
        if module["func"] == "Conv2D":
            x = self.add_convolutional_layer(layer_input, module)
        elif module["func"] == "AvgPool":
            x = self.add_average_pooling_layer(layer_input, module)
        elif module["func"] == "MaxPool":
            x = self.add_max_pooling_layer(layer_input, module)
        elif module["func"] == "Inception":
            x = self.add_inception_layer(layer_input, module)
        return x

    def add_convolutional_layer(self, x, module):
        x = modules.conv_module(x, module["channels"], module["kernel_size"], module["strides"], module["act_func"])
        self.modules.append(x)
        return x

    def add_average_pooling_layer(self, x, module):
        x = AveragePooling2D(module["pool_size"], strides=module["strides"], padding=module["padding"])(x)
        self.modules.append(x)
        return x

    def add_max_pooling_layer(self, x, module):
        x = MaxPooling2D(module["pool_size"], strides=module["strides"], padding=module["padding"])(x)
        self.modules.append(x)
        return x

    def add_inception_layer(self, x, module):
        x = modules.inception_module(x, module["k_1x1"], module["k_3x3"], module["k_5x5"])
        self.modules.append(x)
        return x