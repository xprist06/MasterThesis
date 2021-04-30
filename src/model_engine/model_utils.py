import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Dropout

from model_engine import modules

import numpy as np
import math
import gc



class ModelUtils:
    input = None
    classes_cnt = None

    train_x = None
    train_y = None
    evo_train_x = None
    evo_train_y = None
    test_x = None
    test_y = None
    val_split = None
    verbose = None
    datagen = None
    epochs = None
    batch_size = None

    phase_count = None
    modules_count = None
    phase_output_idx = None
    phase_skip_idx = None

    config = None

    @staticmethod
    def create_model(individual):
        x = modules.conv_module(ModelUtils.input, 32, (3, 3))
        for i in range(ModelUtils.phase_count):
            x = individual.phases[i].create_phase_model(x)
            x = AveragePooling2D((2, 2), strides=(2, 2))(x)
            dropout_val = (i+1)/10
            if dropout_val > 0.5:
                dropout_val = 0.5
            x = Dropout(dropout_val)(x)

        x = modules.fully_connected_module(x, ModelUtils.classes_cnt)
        return Model(ModelUtils.input, x)

    @staticmethod
    def evaluate(queue, individual, validation=False):
        import tensorflow as tf
        import tensorflow.keras as keras
        try:
            sess = tf.compat.v1.Session(config=ModelUtils.config)
            tf.compat.v1.keras.backend.set_session(sess)

            if validation:
                train_x = ModelUtils.train_x
                train_y = ModelUtils.train_y
            else:
                train_x = ModelUtils.evo_train_x
                train_y = ModelUtils.evo_train_y

            model = ModelUtils.create_model(individual)
            optimizer = keras.optimizers.Adam(lr=1e-4, amsgrad=True)
            model.compile(optimizer=optimizer,
                          loss=keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])

            if ModelUtils.datagen is None:
                history = model.fit(
                    train_x,
                    train_y,
                    epochs=ModelUtils.epochs,
                    batch_size=ModelUtils.batch_size,
                    validation_split=ModelUtils.val_split,
                    verbose=ModelUtils.verbose
                )
            else:
                history = model.fit(
                    ModelUtils.datagen.flow(
                        train_x,
                        train_y,
                        batch_size=ModelUtils.batch_size,
                        subset='training'
                    ),
                    validation_data=ModelUtils.datagen.flow(
                        train_x,
                        train_y,
                        batch_size=ModelUtils.batch_size,
                        subset='validation'
                    ),
                    epochs=ModelUtils.epochs,
                    verbose=ModelUtils.verbose
                )

            score = model.evaluate(ModelUtils.test_x, ModelUtils.test_y, verbose=0)

            individual.accuracy = score[1] * 100
            individual.error = (1 - score[1]) * 100
            individual.param_count = np.sum([keras.backend.count_params(w) for w in model.trainable_weights])
            del model
            gc.collect()
            keras.backend.clear_session()
            sess.close()

        except tf.errors.ResourceExhaustedError as e:
            individual.accuracy = 0
            individual.error = 100
            individual.param_count = math.inf
            gc.collect()
            keras.backend.clear_session()
            print("[INFO] Individual can't fit into memory. Skipping...")
        except RuntimeError as e:
            print(e)
            exit(1)
        finally:
            queue.put((individual.accuracy, individual.error, individual.param_count))
