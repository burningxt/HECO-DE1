from src.benchmarks.cec2017 import Cec2017
from src.cython.slow_loops_cy import crossover_exp_cy, crossover_bi_cy, \
    rand_int, rand_normal, rand_cauchy, rand_choice_pb_cy, rand_choice, x_correction, normalization
import numpy as np
import random


class Heco(Cec2017):
    def __init__(self, problem_id, dimension, o_shift, matrix, matrix1, matrix2):
        super().__init__(problem_id, dimension, o_shift, matrix, matrix1, matrix2)
        self.pop_size_init = 12 * self.dimension
        self.lambda_ = 20
        self.H = 5
        self.number_of_strategy = 4
        self.archive_coefficient = 4
        self.gamma = 0.1
        self.fes_max = 20000 * dimension

    def init_pop(self, pop, lb, ub):
        pop[:, :self.dimension] \
            = lb + (ub - lb) * np.random.random_sample((self.pop_size_init, self.dimension))
        for i in range(self.pop_size_init):
            self.evaluate(pop[i, :])

    def init_subpop(self, pop, subpop, pop_size_current):
        selected_indexes = np.random.choice(pop_size_current[0], self.lambda_, replace=False)
        subpop[:, :] = pop[selected_indexes, :]
        return selected_indexes

    def init_memory(self):
        memory_mu = np.full((self.number_of_strategy, self.H), 0.5)
        memory_cr = np.full((self.number_of_strategy, self.H), 0.5)
        return memory_mu, memory_cr

    def generate_mu_cr(self, memory_mu, memory_cr, success_cr, strategy_id):
        mu = rand_cauchy(memory_mu[strategy_id][rand_int(0, self.H - 1)], 0.1)
        cr_ri = memory_cr[strategy_id][rand_int(0, self.H - 1)]
        if len(success_cr[strategy_id]):
            if cr_ri == -1 or max(success_cr[strategy_id]) == 0:
                cr_ri = 0
            cr = rand_normal(cr_ri, 0.1)
        else:
            cr = rand_normal(cr_ri, 0.1)
        while mu <= 0:
            mu = rand_cauchy(memory_mu[strategy_id][rand_int(0, self.H - 1)], 0.1)
        if mu > 1:
            mu = 1
        if cr < 0:
            cr = 0
        elif cr > 1:
            cr = 1
        return mu, cr

    @staticmethod
    def choose_strategy(strategy_ids, strategy_pb, count_success_strategy):
        sum_count = (count_success_strategy + 2.0).sum()
        if sum_count:
            strategy_pb = (count_success_strategy + 2.0) / sum_count
        if strategy_pb[strategy_pb < 0.05].shape[0]:
            strategy_pb[:] = 0.25
            count_success_strategy[:] = 0
        return rand_choice_pb_cy(strategy_ids, strategy_pb)

    def mutation_1(self, subpop, archive, child, mu, archive_size_current, lb, ub, idx):
        x_r1 = rand_int(0, self.lambda_ - 1)
        x_r2 = rand_int(0, self.lambda_ + archive_size_current[0] - 1)
        # self.eq(subpop)
        self.eq_old(subpop)
        best_solution_on_eq = self.find_best(subpop, 1)
        # best_solution_on_obj = subpop[np.lexsort((subpop[:, self.dimension + 2],
        #                                           subpop[:, self.dimension + 3]))][0]
        while x_r1 == idx:
            x_r1 = rand_int(0, self.lambda_ - 1)
        while x_r2 == x_r1 or x_r2 == idx:
            x_r2 = rand_int(0, self.lambda_ + archive_size_current[0] - 1)
        if x_r2 < self.lambda_:
            child[:self.dimension] = subpop[idx, :self.dimension] \
                                     + mu * (best_solution_on_eq[:self.dimension] - subpop[idx, :self.dimension]) \
                                     + mu * (subpop[x_r1, :self.dimension] - subpop[x_r2, :self.dimension])
        else:
            child[:self.dimension] = subpop[idx, :self.dimension]\
                                     + mu * (best_solution_on_eq[:self.dimension] - subpop[idx, :self.dimension]) \
                                     + mu * (subpop[x_r1, :self.dimension]
                                             - archive[x_r2 - self.lambda_, :self.dimension])
        x_correction(child, self.dimension, lb, ub)

    def mutation_2(self, subpop, child, mu, lb, ub):
        x_1, x_2, x_3 = rand_choice(self.lambda_)
        child[:self.dimension] = subpop[x_1, :self.dimension] + mu * (subpop[x_2, :self.dimension]
                                                                      - subpop[x_3, :self.dimension])
        x_correction(child, self.dimension, lb, ub)

    def crossover_exp(self, subpop, child, cr, idx):
        crossover_exp_cy(subpop, child, self.dimension, cr, idx)

    def crossover_bi(self, subpop, child, cr, idx):
        crossover_bi_cy(subpop, child, self.dimension, cr, idx)

    def differential_evolution(self, subpop, archive, child, mu, cr, archive_size_current, strategy_id, lb, ub, idx):
        if strategy_id == 0:
            self.mutation_1(subpop, archive, child, mu, archive_size_current, lb, ub, idx)
            self.crossover_exp(subpop, child, cr, idx)
        elif strategy_id == 1:
            self.mutation_1(subpop, archive, child, mu, archive_size_current, lb, ub, idx)
            self.crossover_bi(subpop, child, cr, idx)
        elif strategy_id == 2:
            self.mutation_2(subpop, child, mu, lb, ub)
            self.crossover_exp(subpop, child, cr, idx)
        elif strategy_id == 3:
            self.mutation_2(subpop, child, mu, lb, ub)
            self.crossover_bi(subpop, child, cr, idx)

    def fitness(self, subpop_plus, fes, idx):
        # self.eq(subpop_plus)
        self.eq_old(subpop_plus)
        equ_norm_idx, obj_norm_idx, vio_norm_idx, equ_norm_child, obj_norm_child, vio_norm_child \
            = normalization(subpop_plus, self.dimension, idx)
        w_t = fes / self.fes_max
        w_i = (idx + 1) / self.lambda_
        w1 = w_t * w_i
        w2 = (1.0 - w_t) * (1.0 - w_i)
        w3 = w_t * w_i + self.gamma
        # w1 = 1.0
        # w2 = 1 - w_t
        # w3 = w_t
        subpop_plus[idx, self.dimension] = w1 * equ_norm_idx + w2 * obj_norm_idx + w3 * vio_norm_idx
        subpop_plus[-1, self.dimension] = w1 * equ_norm_child + w2 * obj_norm_child + w3 * vio_norm_child

    def selection(self, subpop_plus, subpop, archive, fitness_improvements, success_mu, success_cr,
                  mu, cr, strategy_id, count_success_strategy, pop_size_current,
                  archive_position, archive_current_size, idx):
        if subpop_plus[idx, self.dimension] > subpop_plus[-1, self.dimension]:
            success_mu[strategy_id].append(mu)
            success_cr[strategy_id].append(cr)
            fitness_improvements[strategy_id].append(abs(subpop_plus[idx, self.dimension]
                                                     - subpop_plus[-1, self.dimension]))
            if archive_position[0] < self.archive_coefficient * pop_size_current[0] - 1:
                archive[archive_position[0], :] = subpop_plus[idx, :]
                archive_position[0] += 1
                archive_current_size[0] += 1
            else:
                archive[rand_int(0, self.archive_coefficient * pop_size_current[0] - 1), :] \
                    = subpop_plus[idx, :]
                archive_current_size[0] = self.archive_coefficient * pop_size_current[0]
            subpop[idx, :] = subpop_plus[-1, :]
            count_success_strategy[strategy_id] += 1

    def eq(self, pop):
        feasible_solutions = pop[pop[:, self.dimension + 3] == 0.0]
        fea_size = feasible_solutions.shape[0]
        if fea_size:
            f_feasible = feasible_solutions[rand_int(0, fea_size - 1), self.dimension + 2]
            pop[:, self.dimension + 1] = abs(f_feasible - pop[:, self.dimension + 2]) + f_feasible
        else:
            pop[:, self.dimension + 1] = pop[:, self.dimension + 2]

    def eq_old(self, pop):
        best_solution_on_obj = pop[np.lexsort((pop[:, self.dimension + 2], pop[:, self.dimension + 3]))][0]
        pop[:, self.dimension + 1] = abs(pop[:, self.dimension + 2] - best_solution_on_obj[self.dimension + 2])

    def find_best(self, pop, axis):
        return pop[np.argmin(pop[:, self.dimension + axis]), :]

    def update_memory(self, memory_mu, memory_cr, success_mu, success_cr, fitness_improvements,
                      memory_position):
        for strategy_id in range(self.number_of_strategy):
            if len(success_mu[strategy_id]):
                arr_fitness_improvements = np.array(fitness_improvements[strategy_id])
                arr_success_mu = np.array(success_mu[strategy_id])
                arr_success_cr = np.array(success_cr[strategy_id])
                improvements_sum = np.sum(arr_fitness_improvements)
                weights = arr_fitness_improvements / (improvements_sum + 1E-50)
                mean_success_mu = (weights * arr_success_mu**2).sum() / ((weights * arr_success_mu).sum() + 1E-50)
                mean_success_cr = (weights * arr_success_cr).sum()
                memory_mu[strategy_id, memory_position[strategy_id]] = mean_success_mu
                memory_cr[strategy_id, memory_position[strategy_id]] = mean_success_cr
                memory_position[strategy_id] += 1
                if memory_position[strategy_id] > self.H - 1:
                    memory_position[strategy_id] = 0

    def linearly_decrease_pop_size(self, pop, pop_size_current, fes):
        pop_size_next = 0
        if pop_size_current[0] > self.lambda_:
            pop_size_next = round((self.lambda_ - self.pop_size_init) / self.fes_max * fes) \
                            + self.pop_size_init
        if self.lambda_ < pop_size_next < pop_size_current[0]:
            for i in range(pop_size_current[0] - pop_size_next):
                if i:
                    pop[:pop_size_current[0] - 1, :] \
                        = np.delete(pop[:pop_size_current[0], :], rand_int(0, pop_size_current[0] - 1), 0)
                    pop_size_current[0] += -1

    def evolution(self):
        pop = np.zeros((self.pop_size_init, self.dimension + 10))
        subpop = np.zeros((self.lambda_, self.dimension + 10))
        child = np.zeros(self.dimension + 10)
        archive = np.zeros((self.archive_coefficient * self.pop_size_init, self.dimension + 10))
        subpop_plus = np.zeros((self.lambda_ + 1, self.dimension + 10))
        memory_mu, memory_cr = self.init_memory()
        strategy_ids = np.arange(self.number_of_strategy)
        strategy_pb = np.full(self.number_of_strategy, 0.25)
        count_success_strategy = np.zeros(self.number_of_strategy)
        lb, ub = self.get_lb_ub(self.problem_id)
        self.init_pop(pop, lb, ub)
        pop_size_current = [self.pop_size_init]
        archive_size_current = [0]
        fes = self.pop_size_init
        archive_position = [0]
        memory_position = [0] * self.number_of_strategy
        best_obj = pop[0, self.dimension + 2]
        best_vio = pop[0, self.dimension + 3]
        while fes < self.fes_max:
            success_mu = [[] for _ in range(self.number_of_strategy)]
            success_cr = [[] for _ in range(self.number_of_strategy)]
            fitness_improvements = [[] for _ in range(self.number_of_strategy)]
            selected_indexes = self.init_subpop(pop, subpop, pop_size_current)
            for idx in range(self.lambda_):
                strategy_id = self.choose_strategy(strategy_ids, strategy_pb, count_success_strategy)
                mu, cr = self.generate_mu_cr(memory_mu, memory_cr, success_cr, strategy_id)
                self.differential_evolution(subpop, archive, child, mu, cr,
                                            archive_size_current, strategy_id, lb, ub, idx)
                self.evaluate(child)
                subpop_plus[:self.lambda_, :] = subpop
                subpop_plus[self.lambda_, :] = child
                self.fitness(subpop_plus, fes, idx)
                self.selection(subpop_plus, subpop, archive, fitness_improvements, success_mu, success_cr,
                               mu, cr, strategy_id, count_success_strategy, pop_size_current,
                               archive_position, archive_size_current, idx)
                fes += 1
            pop[selected_indexes, :] = subpop
            self.update_memory(memory_mu, memory_cr, success_mu, success_cr, fitness_improvements,
                               memory_position)
            pop[:] = pop[np.lexsort((pop[:, self.dimension + 2], pop[:, self.dimension + 3]))]
            self.linearly_decrease_pop_size(pop, pop_size_current, fes)
            if best_vio > pop[0, self.dimension + 3]:
                best_vio = pop[0, self.dimension + 3]
                best_obj = pop[0, self.dimension + 2]
            elif best_vio == pop[0, self.dimension + 3]:
                if best_obj > pop[0, self.dimension + 2]:
                    best_obj = pop[0, self.dimension + 2]
            best_solution_on_obj = self.find_best(pop[:pop_size_current[0], :], 2)
            print(fes, best_obj, best_vio, pop[0, self.dimension + 2], pop[0, self.dimension + 3],
                  best_solution_on_obj[self.dimension + 2], best_solution_on_obj[self.dimension + 3])
        return best_obj, best_vio



