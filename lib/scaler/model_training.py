import math

from config import *
from lib.scaler.preprocessing_data.data_preprocessor import DataPreprocessor
from lib.includes.utility import *
from lib.scaler.models.ann_model import AnnPredictor
from lib.scaler.models.lstm_model import LstmPredictor
from lib.evolution_algorithms.genetic_algorithm import GenerticAlgorithmEngine
from lib.gaussian_process.gaussian_process import GaussProcess
from lib.evaluation.fitness_manager import FitnessManager
from lib.evolution_algorithms.mfea import MFEAEngine


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

    def build_lstm(self, item=None, cloud_metrics=None, return_data=True):
        if cloud_metrics is None:
            cloud_metrics = {
                'train_data_type': 'mem',
                'predict_data': 'mem'
            }
        
        # Get hyperparameter information
        scaler_method = int(item['scaler'])
        sliding = item['sliding']
        batch_size = item['batch_size']
        num_units = generate_units_size(item['network_size'], item['layer_size'])

        activation = int(item['activation'])
        optimizer = int(item['optimizer'])
        dropout = item['dropout']
        learning_rate = item['learning_rate']

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
        if return_data:
            return lstm_predictor, x_train, y_train, x_test, y_test, data_normalizer
        else:
            return lstm_predictor

    def fit_with_lstm(self, item=None, weights=None, cloud_metrics=None,  fitness_type=None):
        
        # Set up cloud_metrics and fitness_type in case they are None

        if fitness_type is None:
            fitness_type = Config.FITNESS_TYPE

        lstm_predictor, x_train, y_train, x_test, y_test, data_normalizer = self.build_lstm(item, cloud_metrics)

        if weights is None:
            lstm_predictor.fit(x_train, y_train, validation_split=Config.VALID_SIZE,
                            epochs=Config.EPOCHS)

            validation_point = int(Config.VALID_SIZE * x_train.shape[0])
            x_valid = x_train[validation_point:]
            y_valid = y_train[validation_point:]
            fitness = self.fitness_manager.evaluate(lstm_predictor, data_normalizer, x_valid, y_valid)

            return fitness, lstm_predictor
        else:
            # Evaluate weights of the model
            # lstm_predictor.get_weights()
            # model_weights_shape = lstm_predictor.get_model_shape()
            lstm_predictor.set_weights(weights)
            validation_point = int(Config.VALID_SIZE * x_train.shape[0])
            x_valid = x_train[validation_point:]
            y_valid = y_train[validation_point:]
            fitness = self.fitness_manager.evaluate(lstm_predictor, data_normalizer, x_valid, y_valid)
            # print(fitness)
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
        if Config.METHOD_OPTIMIZE == 'bayesian_mfea':
            gauss_process = GaussProcess(self.fit_with_lstm)
            gauss_process.optimize()
        elif Config.METHOD_OPTIMIZE == 'evolutionary_mfea':
            # cloud_metrics = {
            #     'train_data_type': 'mem',
            #     'predict_data': 'mem'
            # }
            item1 = {
                'scaler': 1,
                'sliding': 2,
                'batch_size': 32,
                'network_size': 2,
                'layer_size': 4,
                'activation': 0,
                'optimizer': 0,
                'dropout': 0.2,
                'learning_rate': 3e-4
            }
            lstm_predictor1 = self.build_lstm(item1, return_data=False)
            lstm_shape1 = lstm_predictor1.get_model_shape()
            item2 = {
                'scaler': 1,
                'sliding': 4,
                'batch_size': 32,
                'network_size': 3,
                'layer_size': 8,
                'activation': 0,
                'optimizer': 0,
                'dropout': 0.2,
                'learning_rate': 3e-4
            }
            lstm_predictor2 = self.build_lstm(item2, return_data=False)
            lstm_shape2 = lstm_predictor2.get_model_shape()
            # lstm_weight = lstm_predictor.get_weights()
            # print("LSTM weight: ", lstm_weight)
            # print("LSTM shape: ", lstm_shape)
            # print("Type of LSTM shape: ", type(lstm_shape))
            # print("LSTM shape[0]: ", lstm_shape[0])
            # print("Type of LSTM shape[0]: ", type(lstm_shape[0]))
            # print("Type of shape[0][0]: ", type(lstm_shape[0][0]))
            mfea_engine = MFEAEngine([item1, item2], [lstm_shape1, lstm_shape2], self.fit_with_lstm)
            mfea_engine.evolve()
            # random_weights = get_random_weight(lstm_shape)
            # self.fit_with_lstm(item, weights=random_weights, fitness_type='bayesian_autoscaling')
        else:
            print('[ERROR] METHOD_OPTIMIZE is not supported')

    def train(self):
        print('[3] >>> Start choosing model and experiment')
        if Config.MODEL_EXPERIMENT.lower() == 'ann':
            self.train_with_ann()
        elif Config.MODEL_EXPERIMENT.lower() == 'lstm':
            self.train_with_lstm()
        else:
            print('>>> Can not experiment your method <<<')
        print('[3] >>> Choosing model and experiment complete')
