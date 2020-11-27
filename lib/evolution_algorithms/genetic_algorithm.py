import time
import random

import matplotlib.pyplot as plt
import numpy as np

from lib.includes.utility import *
from config import *


class GenerticAlgorithmEngine:
    cross_over_rate = 0.9
    mutation_rate = 0.05

    def __init__(self, fitness_function):
        self.fitness_function = fitness_function
        self._parse_domain()

        self.population_size = Config.POPULATION_SIZE
    
    def _parse_domain(self):
        domain = Config.LSTM_CONFIG['domain']
        name = []
        type_attr = []
        max_val = []
        min_val = []
        range_val = []

        for attr in domain:
            name.append(attr['name'])
            type_attr.append(attr['type'])
            if attr['type'] == 'discrete':
                min_val.append(0)
                max_val.append(len(attr['domain']) - 1)
            elif attr['type'] == 'continuous':
                min_val.append(attr['domain'][0])
                max_val.append(attr['domain'][1])
            range_val.append(attr['domain'])

        self.name = name
        self.type_attr = type_attr
        self.max_val = np.array(max_val)
        self.min_val = np.array(min_val)
        self.range_val = range_val

    def __str__(self):
        print("My position: {} and pbest is: {}".format(self.pbest_position, self.pbest_value))

    def create_solution(self):
        _solution = self.min_val + (self.max_val - self.min_val) * np.random.rand(len(self.type_attr))
        _solution = self._corect_pos(_solution)
        _fitness, _model = self.evaluate(_solution)
        return [_solution, _fitness, _model]

    def _corect_pos(self, position):
        for i, _type in enumerate(self.type_attr):
            if _type == 'discrete':
                position[i] = int(position[i])
        return position

    def decode_position(self, position):
        result = {}
        for i, _type in enumerate(self.type_attr):
            if _type == 'discrete':
                result[self.name[i]] = self.range_val[i][int(position[i])]
            else:
                result[self.name[i]] = position[i]
        return result
    
    def evaluate(self, position):
        fitness, model = self.fitness_function(self.decode_position(position))
        #print(fitness)
        return fitness, model

    def cal_rank(self, pop):
        '''
        Calculate ranking for element in current population
        '''
        fit = []
        for i in range(len(pop)):
            fit.append(pop[i][1])
        arg_rank = np.array(fit).argsort()
        rank = [i / sum(range(1, len(pop) + 1))
                for i in range(1, len(pop) + 1)]
        return rank

    def wheel_select(self, pop, prob):
        '''
        Select dad and mom from current population by rank
        '''
        r = np.random.random()
        sum = prob[0]
        for i in range(1, len(pop) + 1):
            if sum > r:
                return i - 1
            else:
                sum += prob[i]
        return sum

    def cross_over(self, dad_element, mom_element):
        '''
        crossover dad and mom choose from current population
        '''
        r = np.random.random()
        child1_element = []
        if r < self.cross_over_rate:
            for i in range(len(dad_element[0])):
                n = random.choice([0, 1])
                if n == 0:
                    child1_element.append(dad_element[0][i])
                else:
                    child1_element.append(mom_element[0][i])
            fit1, model1 = self.fitness_function(self.decode_position(child1_element))
            if fit1 < dad_element[1] and fit1 < mom_element[1]:
                return [child1_element, fit1, model1]
            return [child1_element, fit1, model1]
        if dad_element[1] < mom_element[1]:
            return dad_element
        else:
            return mom_element

    def select(self, pop):
        '''
        Select from current population and create new population
        '''
        new_pop = []
        sum_fit = 0
        for i in range(len(pop)):
            sum_fit += pop[0][1]
        while len(new_pop) < self.population_size:
            rank = self.cal_rank(pop)
            dad_index = self.wheel_select(pop, rank)
            mom_index = self.wheel_select(pop, rank)
            while dad_index == mom_index:
                mom_index = self.wheel_select(pop, rank)
            dad = pop[dad_index]
            mom = pop[mom_index]
            new_sol1 = self.cross_over(dad, mom)
            new_pop.append(new_sol1)
        return new_pop

    def mutate(self, pop):
        '''
        Mutate new population
        '''
        for i in range(len(pop)):
            if np.random.random() < self.mutation_rate:
                for j in range(len(pop[i][0])):
                    if np.random.random() < self.mutation_rate:
                        pop[i][0][j] = random.uniform(0, 1)
                pop[i][1], pop[i][2] = self.fitness_function(self.decode_position(pop[i][0]))
        return pop

    def early_stopping(self, array, patience=5):
        if patience <= len(array) - 1:
            value = array[len(array) - patience]
            arr = array[len(array) - patience + 1:]
            check = 0
            for val in arr:
                if val < value:
                    check += 1
            if check != 0:
                return False
            return True
        raise ValueError

    def evolve(self, max_iter, step_save=1):
        print('|-> Start evolve with genertic algorithms')
        pop = [self.create_solution() for _ in range(self.population_size)]
        gbest = pop[0]
        g_best_arr = [gbest[1]]
        print('g_best at epoch 0: {}'.format(gbest[1]))
        for iter in range(max_iter):
            print('Iteration {}'.format(iter + 1))
            start_time = time.time()
            pop = self.select(pop)
            pop = self.mutate(pop)
            best_fit = min(pop, key=lambda x: x[1])
            if best_fit[1] < g_best_arr[-1]:
                gbest = best_fit
            g_best_arr.append(gbest[1])
            print('best current fit {}, best fit so far {}, iter {}'.format(
                best_fit[1], gbest[1], iter))
            print(' Time for running: {}'.format(time.time() - start_time))
        return gbest, np.array(g_best_arr)
