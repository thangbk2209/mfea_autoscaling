
import pickle as pk

from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv

from config import *


class DataReader:
    def __init__(self):
        self.google_trace_config = Config.GOOGLE_TRACE_DATA_CONFIG
        self.normal_data_file = PROJECT_DIR + '/data/input_data/{}/normalized_data.pkl'.format(Config.DATA_EXPERIMENT)

    def __read_google_trace(self):
            time_interval = self.google_trace_config['time_interval']
            data_type = self.google_trace_config['data_type']
            data_name = self.google_trace_config['file_data_name'].format(data_type, time_interval)
            data_file_path = self.google_trace_config['data_path'].format(data_name)

            if Config.GOOGLE_TRACE_DATA_CONFIG['data_type'] == '1_job':
                colnames = ['cpu_rate', 'mem_usage', 'disk_io_time', 'disk_space']
                usecols = [3, 4, 9, 10]
            elif Config.GOOGLE_TRACE_DATA_CONFIG['data_type'] == 'all_jobs':
                colnames = ['cpu_rate', 'mem_usage', 'disk_io_time', 'disk_space']
                usecols = [0, 1, 3, 4]

            google_trace_df = read_csv(
                data_file_path, header=None, index_col=False, names=colnames, usecols=usecols, engine='python')
            cpu = google_trace_df['cpu_rate'].values.reshape(-1, 1)
            mem = google_trace_df['mem_usage'].values.reshape(-1, 1)
            disk_io_time = google_trace_df['disk_io_time'].values.reshape(-1, 1)
            disk_space = google_trace_df['disk_space'].values.reshape(-1, 1)

            official_data = {
                'cpu': cpu,
                'mem': mem,
                'disk_io_time': disk_io_time,
                'disk_space': disk_space
            }
            return official_data

    def read(self):
        if Config.DATA_EXPERIMENT == 'google_trace':
            data = self.__read_google_trace()
            return data
        else:
            print('>>> We do not support to experiment with this data <<<')
            return None, None
