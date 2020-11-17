import math

from config import *
from lib.scaler.preprocessing_data.data_preprocessor import DataPreprocessor
from lib.includes.utility import *
from lib.scaler.models.ann_model import AnnPredictor
from lib.scaler.models.lstm_model import LstmPredictor
from lib.evolution_algorithms.genetic_algorithm import GenerticAlgorithmEngine
from lib.gaussian_process.gaussian_process import GaussProcess
from lib.evaluation.fitness_manager import FitnessManager


class ModelTrainer:
    def __init__(self):
        self.lstm_config = Config.LSTM_CONFIG
        self.ann_config = Config.ANN_CONFIG
        self.results_save_path = Config.RESULTS_SAVE_PATH
        self.fitness_manager = FitnessManager()

    def fit_with_ann(self, item, fitness_type=None):
        scaler_method = item['scaler']
        sliding = item['sliding']
        batch_size = item['batch_size']
        num_units = item['num_units']
        activation = item['activation']
        optimizer = item['optimizer']
        dropout = item['dropout']
        learning_rate = item['learning_rate']

        scaler_method = Config.SCALERS[scaler_method - 1]
        activation = Config.ACTIVATIONS[activation - 1]
        optimizer = Config.OPTIMIZERS[optimizer - 1]

        x_train, y_train, x_test, y_test, data_normalizer = \
            self.data_preprocessor.init_data_ann(sliding, scaler_method)

        input_shape = [x_train.shape[1]]
        output_shape = [y_train.shape[1]]
        model_name = create_name(input_shape=input_shape, output_shape=output_shape, batch_size=batch_size,
                                 num_units=num_units, activation=activation, optimizer=optimizer, dropout=dropout,
                                 learning_rate=learning_rate)

        folder_path = f'{self.results_save_path}models'
        gen_folder_in_path(folder_path)
        model_path = f'{folder_path}/{model_name}'

        ann_predictor = AnnPredictor(
            model_path=model_path,
            input_shape=input_shape,
            output_shape=output_shape,
            batch_size=batch_size,
            num_units=num_units,
            activation=activation,
            optimizer=optimizer,
            dropout=dropout,
            learning_rate=learning_rate
        )

        ann_predictor.fit(x_train, y_train, validation_split=Config.VALID_SIZE,
                          batch_size=batch_size, epochs=Config.EPOCHS)

        fitness = ann_predictor.history.history['val_loss'][-1]

        return fitness, ann_predictor

    def train_with_ann(self):
        item = {
            'scaler': 1,
            'sliding': 2,
            'batch_size': 8,
            'num_units': [4, 2],
            'activation': 1,
            'optimizer': 1,
            'dropout': 0.1,
            'learning_rate': 3e-4
        }
        fitness, ann_predictor = self.fit_with_ann(item)

    def fit_with_lstm(self, item, cloud_metrics=None,  fitness_type=None):
        if cloud_metrics is None:
            cloud_metrics = {
                'train_data_type': 'mem',
                'predict_data': 'mem'
            }

        if fitness_type is None:
            fitness_type = Config.FITNESS_TYPE

        scaler_method = int(item['scaler'])
        sliding = item['sliding']
        batch_size = item['batch_size']
        # num_units = item['num_units']
        num_units = generate_units_size(item['network_size'], item['layer_size'])

        activation = int(item['activation'])
        optimizer = int(item['optimizer'])
        dropout = item['dropout']
        learning_rate = item['learning_rate']

        if type(scaler_method) == int and type(activation) == int and type(optimizer) == int:
            scaler_method = Config.SCALERS[scaler_method - 1]
            activation = Config.ACTIVATIONS[activation - 1]
            optimizer = Config.OPTIMIZERS[optimizer - 1]

        self.data_preprocessor = DataPreprocessor(cloud_metrics)
        x_train, y_train, x_test, y_test, data_normalizer = \
            self.data_preprocessor.init_data_lstm(sliding, scaler_method)

        input_shape = [x_train.shape[1], x_train.shape[2]]
        output_shape = [y_train.shape[1]]
        model_name = create_name(input_shape=input_shape, output_shape=output_shape, batch_size=batch_size,
                                 num_units=num_units, activation=activation, optimizer=optimizer, dropout=dropout,
                                 learning_rate=learning_rate)

        folder_path = f'{self.results_save_path}models'
        gen_folder_in_path(folder_path)
        model_path = f'{folder_path}/{model_name}'

        lstm_predictor = LstmPredictor(
            model_path=model_path,
            input_shape=input_shape,
            output_shape=output_shape,
            batch_size=batch_size,
            num_units=num_units,
            activation=activation,
            optimizer=optimizer,
            dropout=dropout,
            learning_rate=learning_rate
        )

        lstm_predictor.fit(x_train, y_train, validation_split=Config.VALID_SIZE,
                           batch_size=batch_size, epochs=Config.EPOCHS)

        validation_point = int(Config.VALID_SIZE * x_train.shape[0])
        x_valid = x_train[validation_point:]
        y_valid = y_train[validation_point:]
        fitness = self.fitness_manager.evaluate(lstm_predictor, data_normalizer, x_valid, y_valid)

        # elif fitness_type == 'scaler_error':
        #     n_train = int((1 - Config.VALID_SIZE) * len(x_train))
        #     x_valid = x_train[n_train:]
        #     y_valid = y_train[n_train:]
        #     fitness_manager = FitnessManager()
        #     fitness = fitness_manager.evaluate_fitness_scaling(
        #         lstm_predictor, data_normalizer, x_valid, y_valid)
        # else:
        #     print(f'[ERROR] Do not support {fitness_type}')

        return fitness, lstm_predictor

    def train_with_lstm(self):
        # item = {
        #     'scaler': 'min_max_scaler',
        #     'sliding': 4,
        #     'batch_size': 32,
        #     'num_units': [4, 2],
        #     'activation': 'tanh',
        #     'optimizer': 'adam',
        #     'dropout': 0.5,
        #     'learning_rate': 3e-4
        # }
        # self.fit_with_lstm(item, fitness_type='bayesian_autoscaling')
        # genetic_algorithm_ng = GenerticAlgorithmEngine(self.fit_with_lstm)
        # genetic_algorithm_ng.evolve(Config.MAX_ITER)
        gp = GaussProcess(self.fit_with_lstm)
        gp.optimize()

    def train(self):
        print('[3] >>> Start choosing model and experiment')
        if Config.MODEL_EXPERIMENT.lower() == 'ann':
            self.train_with_ann()
        elif Config.MODEL_EXPERIMENT.lower() == 'lstm':
            self.train_with_lstm()
        else:
            print('>>> Can not experiment your method <<<')
        print('[3] >>> Choosing model and experiment complete')
