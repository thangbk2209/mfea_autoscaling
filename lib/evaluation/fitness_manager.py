import math

import numpy as np

from config import *
from lib.evaluation.error_metrics import *

class FitnessManager:
    def __init__(self):
        self.alpha = 0.5

    def _evaluate_validation_error(self, model):
        fitness = model.history.history['val_loss'][-1]
        if math.isnan(fitness):
            fitness = 10
        return fitness

    def _evaluate_scaler_error(self, model, data_normalizer, x_valid, y_valid):
        assert x_valid is not None, '[ERROR] in _evaluate_scaler_error: x_valid is invalid'
        assert y_valid is not None, '[ERROR] in _evaluate_scaler_error:y_valid is invalid'
        assert data_normalizer is not None, '[ERROR] in _evaluate_scaler_error: data_normalizer is invalid'
        validation_error = model.history.history['val_loss'][-1]
        if math.isnan(validation_error):
            return 10

        y_valid_pred = model.predict(x_valid)
        y_valid_pred = data_normalizer.invert_tranform(y_valid_pred)
        y_valid_real = data_normalizer.invert_tranform(y_valid)

        evaluate_validation_prediction = evaluate(y_valid_pred, y_valid_real)
        validation_error = evaluate_validation_prediction['rmse']
        scaling_gap = np.full(y_valid_pred.shape, validation_error)

        upper_valid_scale_value = np.add(y_valid_pred, scaling_gap)
        lower_valid_scaler_value = np.subtract(y_valid_pred, scaling_gap)

        normalized_valid_scale_value = data_normalizer.y_tranform(upper_valid_scale_value)
        scaling_error = evaluate(normalized_valid_scale_value, y_valid)['rmse']
        
        quality_of_service_error = 0
        for i in range(upper_valid_scale_value.shape[0]):
            if y_valid_real[i][0] > upper_valid_scale_value[i][0] or y_valid_real[i][0] < lower_valid_scaler_value[i][0]:
                quality_of_service_error += 1 / upper_valid_scale_value.shape[0]

        return self.alpha * scaling_error + (1-self.alpha * quality_of_service_error)

    def evaluate(self, model=None, data_normalizer=None, x_valid=None, y_valid=None):

        assert model is not None, 'Model should not be None'

        if Config.FITNESS_TYPE == 'validation_error':
            return self._evaluate_validation_error(model)
        elif Config.FITNESS_TYPE == 'scaler_error':
            return self._evaluate_scaler_error(model, data_normalizer, x_valid, y_valid)
        else:
            print(f'ERROR: {Config.FITNESS_TYPE} is not supported')