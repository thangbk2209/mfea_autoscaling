from lib.evolution_algorithms.evolutionary_mfea.Task import Task
from lib.evolution_algorithms.evolutionary_mfea.Flatten import shape_to_dims
from lib.evolution_algorithms.evolutionary_mfea.mfea import MFEA
from lib.evolution_algorithms.evolutionary_mfea.data_mfea import DataMFEA
from config import *

class MFEAEngine:
    def __init__(self, model_item=None, model_shape=None, fitness_function=None):
        self.model_item = model_item
        self.model_shape = model_shape
        self.fitness_function = fitness_function
        self.population_size = Config.POPULATION_SIZE
        self.iterations = 10
        self.MFEAProcess = None

    def init_task(self):
        self.tasks = []
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

        print(data_MFEA.EvBestFitness)
        print(data_MFEA.bestInd_data)
