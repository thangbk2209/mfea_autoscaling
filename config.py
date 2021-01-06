import os

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
CORE_DATA_DIR = PROJECT_DIR + '/{}'.format('data')

ENV_LIST = ['development', 'experiment']
ENV_DEFAULT = 'development'

ENV = ENV_DEFAULT


class Config:
    if ENV == 'development':


        DATA_EXPERIMENT = 'google_trace'  # grid, traffic, google_trace

        GOOGLE_TRACE_DATA_CONFIG = {
            'data_type': 'all_jobs',  # 1_job, all_jobs
            'time_interval': 5,
            'file_data_name': '/input_data/google_trace/{}/{}_mins.csv',
            'data_path': CORE_DATA_DIR + '{}'
        }

        LSTM_CONFIG = {
            'sliding': [5],
            'batch_size': [8],
            'num_units': [[4]],
            'dropout_rate': [0.9],
            'domain': [
                {'name': 'scaler', 'type': 'discrete', 'domain': [1, 2]},
                {'name': 'batch_size', 'type': 'discrete', 'domain': [8, 16, 32, 64, 128]},
                {'name': 'sliding', 'type': 'discrete', 'domain': [2, 3, 4, 5]},
                {'name': 'network_size', 'type': 'discrete', 'domain': [1, 2, 3, 4, 5]},
                {'name': 'layer_size', 'type': 'discrete', 'domain': [4, 8, 16, 32, 64]},
                {'name': 'dropout', 'type': 'continuous', 'domain': (0.0, 0.01)},
                {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.01)},
                {'name': 'optimizer', 'type': 'discrete', 'domain': [1, 2, 3, 4]},
                {'name': 'activation', 'type': 'discrete', 'domain': [1, 2, 3, 4]}
            ]
        }

        ANN_CONFIG = {
            'sliding': [3],
            'batch_size': [8],
            'num_units': [[4]],
            'domain': [
                {'name': 'scaler', 'type': 'discrete', 'domain': [1]},
                {'name': 'batch_size', 'type': 'discrete', 'domain': [64]},
                {'name': 'sliding', 'type': 'discrete', 'domain': [1, 2, 3, 4, 5, 6, 7, 8]},
                {'name': 'network_size', 'type': 'discrete', 'domain': [2, 3, 4, 5]},
                {'name': 'layer_size', 'type': 'discrete', 'domain': [16]},
                {'name': 'dropout', 'type': 'continuous', 'domain': (0.1, 0.5)},
                {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0003, 0.00031)},
                {'name': 'optimizer', 'type': 'discrete', 'domain': [2]},
                {'name': 'activation', 'type': 'discrete', 'domain': [2]}
            ]
        }

        MODEL_EXPERIMENT = 'lstm'  # lstm, ann
        FITNESS_TYPE = 'scaler_error'  # validation_error, scaler_error
        METHOD_OPTIMIZE = 'evolutionary_mfea'  # evolutionary_mfea, bayesian_mfea
        RESULTS_SAVE_PATH = CORE_DATA_DIR + \
            '/{}/{}/'.format(MODEL_EXPERIMENT, FITNESS_TYPE)

        POPULATION_SIZE = 5
        MAX_ITER = 5
        VERBOSE = 2
        LEARNING_RATE = 3e-4
        EPOCHS = 1
        EARLY_STOPPING = True
        PATIENCE = 20
        TRAIN_SIZE = 0.8
        VALID_SIZE = 0.2

        SCALERS = ['min_max_scaler', 'standard_scaler']
        ACTIVATIONS = ['sigmoid', 'tanh', 'relu', 'elu']
        OPTIMIZERS = ['SGD', 'Adam', 'RMSprop', 'Adagrad']

        ALPHA=0.5

    elif ENV == 'experiment':
        DATA_EXPERIMENT = 'google_trace'  # grid, traffic, google_trace

        GOOGLE_TRACE_DATA_CONFIG = {
            # 'train_data_type': 'mem',  # cpu_mem, uni_mem, uni_cpu
            # 'predict_data': 'mem',
            'data_type': 'all_jobs',  # 1_job, all_jobs
            'time_interval': 5,
            'file_data_name': '/input_data/google_trace/{}/{}_mins.csv',
            'data_path': CORE_DATA_DIR + '{}'
        }

        LSTM_CONFIG = {
            'sliding': [5],
            'batch_size': [8],
            'num_units': [[4]],
            'dropout_rate': [0.9],
            'domain': [
                {'name': 'scaler', 'type': 'discrete', 'domain': [1, 2]},
                {'name': 'batch_size', 'type': 'discrete', 'domain': [8, 16, 32, 64, 128]},
                {'name': 'sliding', 'type': 'discrete', 'domain': [2, 3, 4, 5]},
                {'name': 'network_size', 'type': 'discrete', 'domain': [1, 2, 3, 4, 5]},
                {'name': 'layer_size', 'type': 'discrete', 'domain': [4, 8, 16, 32, 64]},
                {'name': 'dropout', 'type': 'continuous', 'domain': (0.0, 0.01)},
                {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.01)},
                {'name': 'optimizer', 'type': 'discrete', 'domain': [1, 2, 3, 4]},
                {'name': 'activation', 'type': 'discrete', 'domain': [1, 2, 3, 4]}
            ]
        }

        ANN_CONFIG = {
            'sliding': [3],
            'batch_size': [8],
            'num_units': [[4]],
            'domain': [
                {'name': 'scaler', 'type': 'discrete', 'domain': [1]},
                {'name': 'batch_size', 'type': 'discrete', 'domain': [64]},
                {'name': 'sliding', 'type': 'discrete', 'domain': [1, 2, 3, 4, 5, 6, 7, 8]},
                {'name': 'network_size', 'type': 'discrete', 'domain': [2, 3, 4, 5]},
                {'name': 'layer_size', 'type': 'discrete', 'domain': [16]},
                {'name': 'dropout', 'type': 'continuous', 'domain': (0.1, 0.5)},
                {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0003, 0.00031)},
                {'name': 'optimizer', 'type': 'discrete', 'domain': [2]},
                {'name': 'activation', 'type': 'discrete', 'domain': [2]}
            ]
        }

        MODEL_EXPERIMENT = 'lstm'  # lstm, ann
        FITNESS_TYPE = 'scaler_error'  # validation_error, scaler_error
        RESULTS_SAVE_PATH = CORE_DATA_DIR + \
            '/{}/{}/'.format(MODEL_EXPERIMENT, FITNESS_TYPE)

        POPULATION_SIZE = 100
        MAX_ITER = 200
        VERBOSE = 0
        LEARNING_RATE = 3e-4
        EPOCHS = 1
        EARLY_STOPPING = True
        PATIENCE = 20
        TRAIN_SIZE = 0.8
        VALID_SIZE = 0.2

        SCALERS = ['min_max_scaler', 'standard_scaler']
        ACTIVATIONS = ['sigmoid', 'tanh', 'relu', 'elu']
        OPTIMIZERS = ['SGD', 'Adam', 'RMSprop', 'Adagrad']

        ALPHA=0.5
    else:
        raise EnvironmentError
