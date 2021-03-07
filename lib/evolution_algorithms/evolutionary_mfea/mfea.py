from lib.evolution_algorithms.evolutionary_mfea.Chromosome import Chromosome
from lib.evolution_algorithms.evolutionary_mfea.data_mfea import DataMFEA
from config import *
import numpy as np
import random
import pickle
import os

class MFEA:
    """
    Class to initialize and execute MFEA 
    """
    def __init__(self, Tasks, pop, gen, selection_process, rmp):
        self.Tasks = Tasks
        self.pop = pop
        self.gen = gen
        self.selection_process = selection_process
        self.rmp = rmp
        self.no_of_tasks = len(self.Tasks)
        self.population = np.array([])
        self.bestobj = np.Inf * np.ones(self.no_of_tasks)
        self.EvBestFitness = np.zeros([self.gen, self.no_of_tasks])
        self.bestInd_data = [None] * self.no_of_tasks
  
                
    def init_population(self):
        self.pop += self.pop % 2
        # if self.no_of_tasks <= 1:
        #     raise Exception("Number of tasks must be at least 2")
        D = np.array([t.dims for t in self.Tasks])
        self.D_multitask = np.max(D)
        
        for i in range(self.pop):
            new_individual = Chromosome(self.D_multitask)
            new_individual.evaluate(self.Tasks, self.no_of_tasks)
            self.population = np.append(self.population, new_individual)
        
        factorial_cost = np.zeros(self.pop)
        for i in range(self.no_of_tasks):
            for j in range(self.pop):
                factorial_cost[j] = self.population[j].factorial_costs[i]
            self.population = self.population[np.argsort(factorial_cost)]
            for j in range(self.pop):
                self.population[j].factorial_ranks[i] = j + 1
            self.bestobj[i] = self.population[0].factorial_costs[i]
            self.EvBestFitness[0][i] = self.bestobj[i]
            self.bestInd_data[i] = self.population[0]
            
            
        for i in range(self.pop):
            min_rank = np.min(self.population[i].factorial_ranks)
            min_rank_skills = np.where(self.population[i].factorial_ranks == min_rank)[0]
            if (np.size(min_rank_skills) > 1):
                self.population[i].skill_factor = min_rank_skills[random.randint(0, np.size(min_rank_skills) - 1)]
            else:
                self.population[i].skill_factor = min_rank_skills[0]
            tmp = self.population[i].factorial_costs[self.population[i].skill_factor]
            self.population[i].factorial_costs[:] = np.Inf
            self.population[i].factorial_costs[self.population[i].skill_factor] = tmp
        
    
    
    def execute(self):
        """
        Execute the MFEA with the given information
        """
        self.init_population()
        generation = 0
        mu = 10 # Index of Simulated Binary Crossover (tunable)
        sigma = 0.02 # standard deviation of Gaussian Mutation model (tunable)
        while generation <= self.gen - 2:
            generation += 1
            print("Generation ", generation)
            indorder = np.random.permutation(self.pop)
            child = np.array([])
            count = 0
            for i in range(0, self.pop//2):
                p1 = indorder[i]
                p2 = indorder[i+self.pop//2]
                child = np.append(child, Chromosome(self.D_multitask))
                child = np.append(child, Chromosome(self.D_multitask))
                if (self.population[p1].skill_factor == self.population[p2].skill_factor \
                or random.random() < self.rmp):
                    u = np.random.rand(self.D_multitask)
                    cf = np.zeros(self.D_multitask)
                    cf[u <= 0.5] = (2 * u[u <= 0.5]) ** (1/(mu+1))
                    cf[u > 0.5] = (2 * u[u > 0.5]) ** (-1/(mu+1))
                    child[count].crossover(self.population[p1], self.population[p2], cf)
                    child[count+1].crossover(self.population[p1], self.population[p2], cf)
                    if random.random() < 0.5:
                        child[count].skill_factor = self.population[p1].skill_factor
                    else:
                        child[count].skill_factor = self.population[p2].skill_factor
                    if random.random() < 0.5:
                        child[count+1].skill_factor = self.population[p1].skill_factor
                    else:
                        child[count+1].skill_factor = self.population[p2].skill_factor
                else:
                    child[count].mutate(self.population[p1], self.D_multitask, sigma)
                    child[count].skill_factor = self.population[p1].skill_factor
                    child[count+1].mutate(self.population[p2], self.D_multitask, sigma)
                    child[count+1].skill_factor = self.population[p2].skill_factor
                count += 2

            for i in range(self.pop):
                child[i].evaluate(self.Tasks, self.no_of_tasks)
            intpopulation = np.concatenate((self.population, child))
            factorial_cost = np.zeros(self.pop * 2)
            for i in range(self.no_of_tasks):
                for j in range(self.pop * 2):
                    factorial_cost[j] = intpopulation[j].factorial_costs[i]
                intpopulation = intpopulation[np.argsort(factorial_cost)]
                for j in range(self.pop * 2):
                    intpopulation[j].factorial_ranks[i] = j + 1
                if (intpopulation[0].factorial_costs[i] <= self.bestobj[i]):
                    self.bestobj[i] = intpopulation[0].factorial_costs[i]
                    self.bestInd_data[i] = intpopulation[0]
                self.EvBestFitness[generation][i] = self.bestobj[i]
            
            for i in range(self.pop * 2):
                min_rank = np.min(intpopulation[i].factorial_ranks)
                intpopulation[i].skill_factor = np.argmin(intpopulation[i].factorial_ranks)
                intpopulation[i].scalar_fitness = 1/min_rank
                
            if self.selection_process == 'elitist':
                # intpopulation.sort(key=lambda ind: ind.scalar_fitness, reverse=True)
                intpopulation = np.array(sorted(list(intpopulation), key = lambda ind: ind.scalar_fitness, reverse = True))
                self.population = intpopulation[:self.pop]
            elif self.selection_process == 'roulette wheel':
                pass

            print("Done with generation ", generation)
            decoded_bestInd_data = self.bestInd_data
            [i.decode(self.Tasks) for i in decoded_bestInd_data]
            print("Fitness for the tasks: ", self.EvBestFitness[generation])
            print("Information of best individual:")
            print(decoded_bestInd_data)
            print("**********************************************************")
            
            currentDir = os.path.dirname(__file__)
            resultDir1 = os.path.join(currentDir, '../../../data/mfea_result/gen_{}/mem'.format(generation))
            resultDir1 = os.path.abspath(os.path.realpath(resultDir1))
            resultDir2 = os.path.join(currentDir, '../../../data/mfea_result/gen_{}/cpu'.format(generation))
            resultDir2 = os.path.abspath(os.path.realpath(resultDir2))
            
            if Config.RUN_OPTION == 12:
                with open(resultDir1, 'wb') as fp:
                    pickle.dump(decoded_bestInd_data[0], fp)
                with open(resultDir2, 'wb') as fp:
                    pickle.dump(decoded_bestInd_data[1], fp)
            elif Config.RUN_OPTION == 1:
                with open(resultDir1, 'wb') as fp:
                    pickle.dump(decoded_bestInd_data[0], fp)
            elif Config.RUN_OPTION == 2:
                with open(resultDir2, 'wb') as fp:
                    pickle.dump(decoded_bestInd_data[0], fp)
                
            # load lai du lieu
            # with open(resultDir1, 'rb') as fp:
            #     decoded_bestInd_data[0] = pickle.load(fp)

            
        data_MFEA = DataMFEA(self.EvBestFitness, self.bestInd_data)
        return data_MFEA