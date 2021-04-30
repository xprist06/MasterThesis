phase_layers = [
    {
        "id": 1,
        "func": "Conv2D",
        "act_func": "relu",
        "channels": 32,
        "kernel_size": (3, 3),
        "strides": (1, 1),
        "desc": "conv_3x3_32"
    },  # conv_3x3_32
    {
        "id": 2,
        "func": "Conv2D",
        "act_func": "relu",
        "channels": 64,
        "kernel_size": (3, 3),
        "strides": (1, 1),
        "desc": "conv_3x3_64"
    },  # conv_3x3_64
    {
        "id": 3,
        "func": "Conv2D",
        "act_func": "relu",
        "channels": 128,
        "kernel_size": (3, 3),
        "strides": (1, 1),
        "desc": "conv_3x3_128"
    },  # conv_3x3_128
    {
        "id": 4,
        "func": "Inception",
        "act_func": "relu",
        "k_1x1": 32,
        "k_3x3": 32,
        "k_5x5": 32,
        "desc": "inc_32"
    },  # inc_32
    {
        "id": 5,
        "func": "Inception",
        "act_func": "relu",
        "k_1x1": 64,
        "k_3x3": 64,
        "k_5x5": 64,
        "desc": "inc_64"
    },  # inc_64
    {
        "id": 6,
        "func": "AvgPool",
        "pool_size": (3, 3),
        "strides": (1, 1),
        "padding": "same",
        "desc": "avg_3x3"
    },  # avg_3x3
    {
        "id": 7,
        "func": "Conv2D",
        "act_func": "relu",
        "channels": 32,
        "kernel_size": (5, 5),
        "strides": (1, 1),
        "desc": "conv_5x5_32"
    },  # conv_5x5_32
    {
        "id": 8,
        "func": "Conv2D",
        "act_func": "relu",
        "channels": 64,
        "kernel_size": (5, 5),
        "strides": (1, 1),
        "desc": "conv_5x5_64"
    },  # conv_5x5_64
    {
        "id": 9,
        "func": "Conv2D",
        "act_func": "relu",
        "channels": 128,
        "kernel_size": (5, 5),
        "strides": (1, 1),
        "desc": "conv_5x5_128"
    },  # conv_5x5_128
{
        "id": 10,
        "func": "MaxPool",
        "pool_size": (3, 3),
        "strides": (1, 1),
        "padding": "same",
        "desc": "max_3x3"
    },  # max_3x3
]
