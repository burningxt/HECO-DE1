from src.heco_de1.heco import Heco
from src.benchmarks.cec2017 import load_mat
import timeit
import multiprocessing as mp


def run(runs):
    for dimension in [10, 30, 50, 100]:
        for problem_id in range(1, 29):
            o, m, m1, m2 = load_mat(problem_id, dimension)
            results = Heco(problem_id, dimension, o, m, m1, m2).evolution()
            print(results[0], file=open("./data_analysis/output_data/F{}_{}D_obj.txt".format(problem_id, dimension), "a"))
            print(results[1], file=open("./data_analysis/output_data/F{}_{}D_vio.txt".format(problem_id, dimension), "a"))


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    start = timeit.default_timer()
    pool = mp.Pool(processes=25)
    res = pool.map(run, range(25))
    stop = timeit.default_timer()
    print('Time: ', stop - start)


# if __name__ == '__main__':
#     start = timeit.default_timer()
#     prob_id = 1
#     dim = 100
#     o, m, m1, m2 = load_mat(prob_id, dim)
#     Heco(prob_id, dim, o, m, m1, m2).evolution()
#     stop = timeit.default_timer()
#     print('Time: ', stop - start)


# print(fes, best_obj, best_vio, pop[0, self.dimension + 2], pop[0, self.dimension + 3],
#                   best_solution_on_obj[self.dimension + 2], best_solution_on_obj[self.dimension + 3],
#                   pop.shape[0])

