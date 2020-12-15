class MFEAEngine:
    def __init__(self, model_shape=None, fitness_function=None):
        self.model_shape = model_shape
        self.fitness_function = fitness_function

    def init_task(self):
        pass

    def create_population(self):
        pass

    def evolve(self, iterations=100):
        self.init_task()
        self.create_population()
        for _iter in range(iterations):
            pass

