import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

from lib.evaluation.error_metrics import evaluate
from lib.includes.utility import *


class BaseModel:
    def __init__(self, model_path, initial_state=True):
        self.model_path = model_path
        if initial_state:
            self._build_model()

    def _build_model(self):
        pass

    def load_model(self, model_path=None):

        if model_path is None:
            self.model = load_model(self.model_path)
        else:
            self.model = load_model(model_path)

    def save_model(self, model_saved_path=None):

        if model_saved_path is None:
            # Save model by self.model_path
            self.model.save(self.model_path)
        else:
            # Save model by model_saved_path
            self.model.save(model_saved_path)

    def get_model_description(self, infor_path):

        try:
            self.model.summary()
            plot_model(self.model, infor_path, show_shapes=True)
        except Exception as ex:
            print('[ERROR] Can not get description of the model')

    def get_model_shape(self):
        results = []
        for _layer in self.model.layers:
            _result = []
            for _weight in _layer.trainable_weights:
                _result.append(tf.keras.backend.get_value(tf.shape(_weight)))
            results.append(_result)
        return results

    def get_weights(self):
        weights = []
        for _layer in self.model.layers:
            _weights = []
            for _weight in _layer.trainable_weights:
                _weights.append(tf.keras.backend.get_value(_weight))
            weights.append(_weights)

            # _weight.assign(value)
                # value = np.random.rand(1, 16)
                # value = tf.random.uniform((1, 16), minval=-1, maxval=1)
                # print('=== value: ', value, value.dtype)
                # print('=== _weight: ', _weight)
                # print(tf.keras.backend.get_value(_weight))
                # print('------')
                # _weight.assign(value)
                # print(tf.keras.backend.get_value(_weight))
                # exit(0)
        #     print(_layer.trainable_weights)
        #     print(_layer.trainable_weights.__dir__())
        #     print('=========')
        # print(self.model.layers)
        return weights

    def set_weights(self, weights):
        for i, _layer in enumerate(self.model.layers):
            for j, _weight in enumerate(_layer.trainable_weights):
                _weight.assign(weights[i][j])

    def plot_learning_curves(self):
        try:
            # plot learning curves
            plt.title('Learning Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.plot(self.history.history['loss'], label='train')
            plt.plot(self.history.history['val_loss'], label='val')
            plt.legend()
            plt.show()
        except Exception as ex:
            print('[ERROR] Can not plot learning curves of the model')

    def fit(self, x, y, validation_split=0, batch_size=1, epochs=10, early_stopping=True, patience=20):
        callbacks = []

        if early_stopping:
            es = EarlyStopping(monitor='val_loss', patience=Config.PATIENCE)
            model_checkpoint = ModelCheckpoint(f'{self.model_path}.h5', monitor='val_loss', mode='min',
                                               verbose=Config.VERBOSE, save_best_only=True)
            callbacks = [es, model_checkpoint]

        self.history = self.model.fit(
            x, y, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=Config.VERBOSE,
            shuffle=False, callbacks=callbacks)
        if os.path.exists(f'{self.model_path}.h5'):
            self.model.load_weights(f'{self.model_path}.h5')

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y, data_normalizer):
        pred = self.predict(x)
        pred = data_normalizer.invert_tranform(pred)

        return evaluate(y, pred, ('mae', 'rmse', 'mse', 'mape', 'smape'))
