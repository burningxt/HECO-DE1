from src.cython.slow_loops_cy import problem_18_27, problem_17_26, np_max
# from src.heco_de1.utils import sgn
from numpy import sin, sqrt, cos, pi, exp, e
# import random
import numpy as np
import scipy.io as sio
import os
script_dir = os.path.dirname(__file__)


def load_mat(problem_id, dimension):
    o_shift = np.zeros((1, dimension), dtype=np.float64)
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    matrix1 = np.zeros((dimension, dimension), dtype=np.float64)
    matrix2 = np.zeros((dimension, dimension), dtype=np.float64)
    matrix_d = {10: 'M_10', 30: 'M_30', 50: 'M_50', 100: 'M_100'}
    matrix1_d = {10: 'M1_10', 30: 'M1_30', 50: 'M1_50', 100: 'M1_100'}
    matrix2_d = {10: 'M2_10', 30: 'M2_30', 50: 'M2_50', 100: 'M2_100'}
    if problem_id in [1, 3, 4, 6, 7, 8, 9]:
        mat_contents = sio.loadmat(script_dir + '/input_data/Function{}.mat'.format(problem_id))
        o_shift[0, :] = mat_contents['o'][0, :dimension]
    elif problem_id == 2:
        mat_contents = sio.loadmat(script_dir + '/input_data/Function2.mat')
        o_shift[0, :] = mat_contents['o'][0, :dimension]
        matrix[:] = mat_contents[matrix_d[dimension]]
    elif problem_id == 5:
        mat_contents = sio.loadmat(script_dir + '/input_data/Function5.mat')
        o_shift[0, :] = mat_contents['o'][0, :dimension]
        matrix1[:] = mat_contents[matrix1_d[dimension]]
        matrix2[:] = mat_contents[matrix2_d[dimension]]
    elif problem_id in range(10, 21):
        mat_contents = sio.loadmat(script_dir + '/input_data/ShiftAndRotation.mat')
        o_shift[0, :] = mat_contents['o'][0, :dimension]
    elif problem_id in range(21, 29):
        mat_contents = sio.loadmat(script_dir + '/input_data/ShiftAndRotation.mat')
        matrix[:] = mat_contents[matrix_d[dimension]]
    return o_shift, matrix, matrix1, matrix2


class Cec2017:
    def __init__(self, problem_id, dimension, o_shift, matrix, matrix1, matrix2):
        self.problem_id = problem_id
        self.dimension = dimension
        self.o_shift = o_shift
        self.matrix = matrix
        self.matrix1 = matrix1
        self.matrix2 = matrix2

    def shift_func(self, x):
        y = x - self.o_shift[0]
        return y

    @staticmethod
    def rotate_func(y, matrix):
        z = np.dot(matrix, y)
        return z

    def benchmark(self, x):
        v, v_g, v_h = (0.0, 0.0, 0.0)
        y = self.shift_func(x)
        if self.problem_id == 1:
            f = ((np.cumsum(y))**2).sum()
            v_g = max(0, (y**2 - 5000.0 * cos(0.1 * pi * y) - 4000.0).sum())
            len_g = 1
            len_h = 0

        elif self.problem_id == 2:
            z = self.rotate_func(y, self.matrix)
            f = ((np.cumsum(y)) ** 2).sum()
            v_g = max(0, (z ** 2 - 5000.0 * cos(0.1 * pi * z) - 4000.0).sum())
            len_g = 1
            len_h = 0

        elif self.problem_id == 3:
            f = ((np.cumsum(y)) ** 2).sum()
            v_g = max(0, (y ** 2 - 5000.0 * cos(0.1 * pi * y) - 4000.0).sum())
            v_h += max(0, abs(y * sin(0.1 * pi * y)).sum() - 1E-4)
            len_g = 1
            len_h = 1

        elif self.problem_id == 4:
            f = (y**2 - 10.0 * cos(2.0 * pi * y) + 10.0).sum()
            v_g += max(0, -(y * sin(2.0 * y)).sum())
            v_g += max(0, (y * sin(y)).sum())
            len_g = 2
            len_h = 0

        elif self.problem_id == 5:
            w1 = self.rotate_func(y, self.matrix1)
            w2 = self.rotate_func(y, self.matrix2)
            f = (100.0 * (y[:-1]**2 - y[1:])**2
                 + (y[:self.dimension - 1] - 1.0)**2).sum()
            v_g += max(0, (w1**2 - 50.0 * cos(2.0 * pi * w1) - 40.0).sum())
            v_g += max(0, (w2**2 - 50.0 * cos(2.0 * pi * w2) - 40.0).sum())
            len_g = 2
            len_h = 0

        elif self.problem_id == 6:
            f = (y**2 - 10.0 * cos(2.0 * pi * y) + 10.0).sum()
            v_h += max(0, abs(-y * sin(2.0 * y)).sum() - 1E-4)
            v_h += max(0, abs(y * sin(2.0 * pi * y)).sum() - 1E-4)
            v_h += max(0, abs(-y * cos(2.0 * y)).sum() - 1E-4)
            v_h += max(0, abs(y * cos(2.0 * pi * y)).sum() - 1E-4)
            v_h += max(0, abs(y * sin(2.0 * (abs(y))**0.5)).sum() - 1E-4)
            v_h += max(0, abs(-y * sin(2.0 * (abs(y)) ** 0.5)).sum() - 1E-4)
            len_g = 0
            len_h = 6

        elif self.problem_id == 7:
            f = (y * sin(y)).sum()
            v_h += max(0, abs(y - 100.0 * cos(0.5 * y) + 100.0).sum() - 1E-4)
            v_h += max(0, abs(-(y - 100.0 * cos(0.5 * y) + 100.0)).sum() - 1E-4)
            len_g = 0
            len_h = 2

        elif self.problem_id == 8:
            f = np_max(y)
            y_odd = y[::2]
            y_even = y[1::2]
            v_h += max(0, abs(np.cumsum(y_odd) ** 2).sum() - 1E-4)
            v_h += max(0, abs(np.cumsum(y_even) ** 2).sum() - 1E-4)
            len_g = 0
            len_h = 2

        elif self.problem_id == 9:
            f = np_max(y)
            y_odd = y[::2]
            y_even = y[1::2]
            v_g += max(0, abs(np.prod(y_even)))
            v_h += max(0, abs((y_odd[:-1] ** 2 - y_odd[1:]) ** 2).sum() - 1E-4)
            len_g = 1
            len_h = 1

        elif self.problem_id == 10:
            f = np_max(y)
            v_h += max(0, abs((np.cumsum(y) ** 2).sum()) - 1E-4)
            v_h += max(0, abs(((y[:-1] - y[1:]) ** 2).sum()) - 1E-4)
            len_g = 0
            len_h = 2

        elif self.problem_id == 11:
            f = y.sum()
            v_g += max(0, abs(np.prod(y)))
            v_h += max(0, abs(((y[:-1] - y[1:]) ** 2).sum()) - 1E-4)
            len_g = 1
            len_h = 1

        elif self.problem_id == 12:
            f = (y ** 2 - 10.0 * cos(2.0 * pi * y) + 10.0).sum()
            v_g += max(0, 4.0 - abs(y).sum())
            v_g += max(0, (y ** 2).sum() - 4.0)
            len_g = 2
            len_h = 0

        elif self.problem_id == 13:
            f = (100.0 * (y[:-1] ** 2 - y[1:]) ** 2 + (y[:-1] - 1.0) ** 2).sum()
            v_g += max(0, (y ** 2 - 10.0 * cos(2.0 * pi * y) + 10.0).sum() - 100.0)
            v_g += max(0, y.sum() - 2.0 * self.dimension)
            v_g += max(0, 5.0 - y.sum())
            len_g = 3
            len_h = 0

        elif self.problem_id == 14:
            f = -20.0 * exp(-0.2 * sqrt(1.0 / self.dimension * (y**2).sum())) + 20.0 \
                - exp(1.0 / self.dimension * (cos(2.0 * pi * y)).sum()) + e
            v_g += max(0, (y[1:]**2).sum() + 1.0 - abs(y[0]))
            v_h += max(0, abs((y**2).sum() - 4.0) - 1E-4)
            len_g = 1
            len_h = 1

        elif self.problem_id == 15:
            f = max(abs(y))
            v_g += max(0, (y**2).sum() - 100.0 * self.dimension)
            v_h += max(0, abs(cos(f) + sin(f)) - 1E-4)
            len_g = 1
            len_h = 1

        elif self.problem_id == 16:
            f = abs(y).sum()
            v_g += max(0, (y**2).sum() - 100.0 * self.dimension)
            v_h += max(0, (cos(f) + sin(f))**2 - exp(cos(f) + sin(f) - 1.0 + exp(1.0)) - 1E-4)
            len_g = 1
            len_h = 1

        elif self.problem_id == 17:
            f, g1 = problem_17_26(self.dimension, y)
            v_g += max(0, abs(1.0 - g1) - 1E-4)
            v_h += max(0, abs((y**2).sum() - 4.0 * self.dimension) - 1E-4)
            len_g = 1
            len_h = 1

        elif self.problem_id == 18:
            z = np.zeros(self.dimension)
            problem_18_27(self.dimension, z, y)
            f = (z**2 - 10.0 * cos(2.0 * pi * z) + 10).sum()
            v_g += max(0, 1 - abs(y).sum())
            v_g += max(0, (y**2).sum() - 100.0 * self.dimension)
            v_h += max(0, abs((100.0 * (y[:-1]**2 - y[1:])**2).sum() + np.prod((sin(y - 1)**2 * pi))) - 1E-4)
            len_g = 2
            len_h = 1

        elif self.problem_id == 19:
            f = (abs(y)**0.5 + 2.0 * sin(y)**3).sum()
            v_g += max(0, (-10.0 * exp(-0.2 * sqrt(y[:-1]**2 + y[1:]**2))).sum()
                       + (self.dimension - 1.0) * 10.0 / exp(-5.0))
            v_g += max(0, (sin(2.0 * y)**2).sum() - 0.5 * self.dimension)
            len_g = 2
            len_h = 0

        elif self.problem_id == 20:
            f = (0.5 + (sin(sqrt(y[:-1]**2 + y[1:]**2))**2 - 0.5)
                 / (1.0 + 0.001 * sqrt(y[:-1]**2 + y[1:]**2))**2).sum() \
                + 0.5 + (sin(sqrt(y[-1]**2 + y[1]**2))**2 - 0.5) / (1.0 + 0.001 * sqrt(y[-1]**2 + y[1]**2))**2
            v_g += max(0, cos(y.sum())**2 - 0.25 * cos(y.sum()) - 0.125)
            v_g += max(0, exp(cos(y.sum())) - exp(0.25))
            len_g = 2
            len_h = 0

        elif self.problem_id == 21:
            z = self.rotate_func(y, self.matrix)
            f = (z**2 - 10.0 * cos(2.0 * pi * z) + 10.0).sum()
            v_g += max(0, 4 - abs(z).sum())
            v_g += max(0, (z**2).sum() - 4.0)
            len_g = 2
            len_h = 0

        elif self.problem_id == 22:
            z = self.rotate_func(y, self.matrix)
            f = (100.0 * (z[:-1]**2 - z[1:])**2 + (z[:-1] - 1)**2).sum()
            v_g += max(0, (z**2 - 10.0 * cos(2.0 * pi * z) + 10.0).sum() - 100.0)
            v_g += max(0, z.sum() - 2.0 * self.dimension)
            v_g += max(0, 5.0 - z.sum())
            len_g = 3
            len_h = 0

        elif self.problem_id == 23:
            z = self.rotate_func(y, self.matrix)
            f = -20.0 * exp(-0.2 * sqrt(1.0 / self.dimension * (z**2).sum())) + 20.0 \
                - exp(1.0 / self.dimension * cos(2.0 * pi * z).sum()) + e
            v_g += max(0, (z[1:]**2).sum() + 1 - abs(z[0]))
            v_h += max(0, abs((z**2).sum() - 4.0) - 1E-4)
            len_g = 1
            len_h = 0

        elif self.problem_id == 24:
            z = self.rotate_func(y, self.matrix)
            f = max(abs(z))
            v_g += max(0, (z**2).sum() - 100.0 * self.dimension)
            v_h += max(0, abs(cos(f) + sin(f)) - 1E-4)
            len_g = 1
            len_h = 1

        elif self.problem_id == 25:
            z = self.rotate_func(y, self.matrix)
            f = abs(z).sum()
            v_g += max(0, (z**2).sum() - 100.0 * self.dimension)
            v_h += max(0, abs((cos(f) + sin(f))**2 - exp(cos(f) + sin(f)) - 1.0 + exp(1.0)) - 1E-4)
            len_g = 1
            len_h = 1

        elif self.problem_id == 26:
            z = self.rotate_func(y, self.matrix)
            f, g1 = problem_17_26(self.dimension, z)
            v_g += max(0, abs(1.0 - g1) - 1E-4)
            v_h += max(0, abs((z ** 2).sum() - 4.0 * self.dimension) - 1E-4)
            len_g = 1
            len_h = 1

        elif self.problem_id == 27:
            z = self.rotate_func(y, self.matrix)
            problem_18_27(self.dimension, y, z)
            f = (y ** 2 - 10.0 * cos(2.0 * pi * y) + 10).sum()
            v_g += max(0, 1 - abs(z).sum())
            v_g += max(0, (z ** 2).sum() - 100.0 * self.dimension)
            v_h += max(0, abs((100.0 * (z[:-1] ** 2 - z[1:]) ** 2).sum() + np.prod((sin(z - 1) ** 2 * pi))) - 1E-4)
            len_g = 2
            len_h = 1

        elif self.problem_id == 28:
            z = self.rotate_func(y, self.matrix)
            f = (abs(z) ** 0.5 + 2.0 * sin(z) ** 3).sum()
            v_g += max(0, (-10.0 * exp(-0.2 * sqrt(z[:-1] ** 2 + z[1:] ** 2))).sum()
                       + (self.dimension - 1.0) * 10.0 / exp(-5.0))
            v_g += max(0, (sin(2.0 * z) ** 2).sum() - 0.5 * self.dimension)
            len_g = 2
            len_h = 0

        v = (v_g + v_h) / (len_g + len_h)
        return f, v, v_g, v_h

    @staticmethod
    def get_lb_ub(problem_id):
        if problem_id in range(1, 4) or problem_id in [8] \
                or problem_id in range(10, 19) or problem_id in range(20, 28):
            lb = -100.0
            ub = 100.0
        elif problem_id in [4, 5, 9]:
            lb = -10.0
            ub = 10.0
        elif problem_id in [6]:
            lb = -20.0
            ub = 20.0
        elif problem_id in [7, 19, 28]:
            lb = -50.0
            ub = 50.0
        return lb, ub

    def evaluate(self, individual):
        results = self.benchmark(individual[:self.dimension])
        individual[self.dimension + 2: self.dimension + 6] = results
