from lib.evolution_algorithms.evolutionary_mfea.Task import Task
from lib.evolution_algorithms.evolutionary_mfea.Flatten import shape_to_dims
from lib.evolution_algorithms.evolutionary_mfea.mfea import MFEA
from lib.evolution_algorithms.evolutionary_mfea.data_mfea import DataMFEA
from config import *
import matplotlib.pyplot as plt
import numpy as np
import os

class MFEAEngine:
    def __init__(self, model_item=None, model_shape=None, fitness_function=None):
        self.model_item = model_item
        self.model_shape = model_shape
        self.fitness_function = fitness_function
        self.population_size = Config.POPULATION_SIZE
        self.iterations = Config.MAX_ITER
        self.MFEAProcess = None

    def init_task(self):
        self.tasks = []
        if Config.RUN_OPTION == 1:
            self.tasks.append(Task('mem', item=self.model_item[0], fnc=self.fitness_function, model_shape=self.model_shape[0]))
        elif Config.RUN_OPTION == 2:
            self.tasks.append(Task('cpu', item=self.model_item[0], fnc=self.fitness_function, model_shape=self.model_shape[0]))
        elif Config.RUN_OPTION == 11:
            self.tasks.append(Task('mem', item=self.model_item[0], fnc=self.fitness_function, model_shape=self.model_shape[0]))
            self.tasks.append(Task('mem', item=self.model_item[0], fnc=self.fitness_function, model_shape=self.model_shape[0]))
        elif Config.RUN_OPTION == 22:
            self.tasks.append(Task('cpu', item=self.model_item[1], fnc=self.fitness_function, model_shape=self.model_shape[1]))
            self.tasks.append(Task('cpu', item=self.model_item[1], fnc=self.fitness_function, model_shape=self.model_shape[1]))
        elif Config.RUN_OPTION == 12:
            self.tasks.append(Task('mem', item=self.model_item[0], fnc=self.fitness_function, model_shape=self.model_shape[0]))
            self.tasks.append(Task('cpu', item=self.model_item[1], fnc=self.fitness_function, model_shape=self.model_shape[1]))

    def create_population(self):
        self.MFEAProcess = MFEA(Tasks=self.tasks,
                                pop=self.population_size,
                                gen=self.iterations,
                                selection_process='elitist',
                                rmp=0.3)
        self.MFEAProcess.init_population()

    def evolve(self):
        self.init_task()
        self.create_population()
        data_MFEA = self.MFEAProcess.execute()
        [i.decode(self.tasks) for i in data_MFEA.bestInd_data]

        print('=== data_MFEA.EvBestFitness ===')
        print(data_MFEA.EvBestFitness)
        
        currentDir = os.path.dirname(__file__)
        # resultDir = os.path.join(currentDir, '../../data/mfea_result/best_fitness.npy')
        if Config.RUN_OPTION == 1:
            resultDir = os.path.join(currentDir, '../../data/mfea_result/mem/best_fitness.npy')
        elif Config.RUN_OPTION == 2:
            resultDir = os.path.join(currentDir, '../../data/mfea_result/cpu/best_fitness.npy')
        elif Config.RUN_OPTION == 11:
            resultDir = os.path.join(currentDir, '../../data/mfea_result/mem_mem/best_fitness.npy')
        elif Config.RUN_OPTION == 22:
            resultDir = os.path.join(currentDir, '../../data/mfea_result/cpu_cpu/best_fitness.npy')
        elif Config.RUN_OPTION == 12:
            resultDir = os.path.join(currentDir, '../../data/mfea_result/mem_cpu/best_fitness.npy')

        resultDir = os.path.abspath(os.path.realpath(resultDir))
        print(resultDir)
        with open(resultDir, mode='wb') as f:
            np.save(f, data_MFEA.EvBestFitness)
            
        with open(resultDir, mode='rb') as f:
            arr = np.load(f)
            if Config.RUN_OPTION > 10:
                if Config.RUN_OPTION // 10 == 1:
                    lb1 = 'mem1'
                else:
                    lb1 = 'cpu1'
                    
                if Config.RUN_OPTION % 10 == 1:
                    lb2 = 'mem2'
                else:
                    lb2 = 'cpu2'
                plt.plot(np.arange(arr.shape[0]), arr[:, 0], label=lb1)
                plt.plot(np.arange(arr.shape[0]), arr[:, 1], label=lb2)
            else: 
                if Config.RUN_OPTION % 10 == 1:
                    lb = 'mem'
                else:
                    lb = 'cpu'
                plt.plot(np.arange(arr.shape[0]), arr[:, 0], label=lb)
            plt.title('Loss function of LSTM network with MFEA optimizer')
            plt.xlabel('Generation')
            plt.ylabel('Loss value')
            plt.legend()
            plt.show()
        
