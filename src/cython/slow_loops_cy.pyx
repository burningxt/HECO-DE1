from libc.math cimport cos, sin, pi, round, fabs, sqrt, log, tan
from libc.stdlib cimport rand
import random
import numpy as np


cdef int _rand_int(int r_min, int r_max):
    cdef:
        int the_int
    the_int = rand() % (r_max - r_min + 1) + r_min
    return the_int

def rand_int(r_min, r_max):
    return _rand_int(r_min, r_max)


cdef double _rand_normal(double mu, double sigma):
    cdef:
        double uniform, z
    uniform = random.random()
    z = sqrt(- 2.0 * log(uniform)) * sin(2.0 * pi * uniform)
    z = mu + sigma * z
    return z

def rand_normal(mu, sigma):
    return _rand_normal(mu, sigma)


cdef double _rand_cauchy(double mu, double gamma):
    cdef:
        double uniform, z
    uniform = random.random()
    z = mu + gamma * tan(pi * (uniform - 0.5))
    return z

def rand_cauchy(mu, gamma):
    return _rand_cauchy(mu, gamma)


cdef double _sgn(double v):
    cdef:
        double sgn_value
    sgn_value = -1.0
    if v > 0.0:
        sgn_value = 1.0
    elif v == 0.0:
        sgn_value = 0.0
    return sgn_value

cdef double _np_max(double[:] z):
    cdef:
        double max_value
        Py_ssize_t bound
    max_value = z[0]
    bound = z.shape[0]
    for i in range(bound):
        if max_value < z[i]:
            max_value = z[i]
    return max_value

def np_max(z):
    return _np_max(z)

cdef tuple _np_max_min(double[:, :] z, int dimension, int axis):
    cdef:
        double max_value, min_value
        Py_ssize_t bound
    max_value = z[0, dimension + axis]
    min_value = z[0, dimension + axis]
    bound = z.shape[0]
    for i in range(bound):
        if max_value < z[i, dimension + axis]:
            max_value = z[i, dimension + axis]
    for i in range(bound):
        if min_value > z[i, dimension + axis]:
            min_value = z[i, dimension + axis]
    return  min_value, max_value

def np_max_min(z, dimension, axis):
    return _np_max_min(z, dimension, axis)


cdef tuple _normalization(double[:, :]subpop_plus, int dimension,  int idx):
    cdef:
        double equ_min, equ_max, obj_min, obj_max, vio_min, \
            vio_max, equ_norm, obj_norm, vio_norm
    equ_min, equ_max = _np_max_min(subpop_plus, dimension, 1)
    obj_min, obj_max = _np_max_min(subpop_plus, dimension, 2)
    vio_min, vio_max = _np_max_min(subpop_plus, dimension, 3)
    equ_norm = (subpop_plus[idx, dimension + 1] - equ_min) / (equ_max - equ_min + 1E-50)
    obj_norm = (subpop_plus[idx, dimension + 2] - obj_min) / (obj_max - obj_min + 1E-50)
    vio_norm = (subpop_plus[idx, dimension + 3] - vio_min) / (vio_max - vio_min + 1E-50)
    return equ_norm, obj_norm, vio_norm

def normalization(subpop_plus, dimension, idx):
    return _normalization(subpop_plus, dimension, idx)


cdef void _problem_18_27(int dimension, double[:]y, double[:]z):
    for i in range(dimension):
        if fabs(z[i]) < 0.5:
            y[i] = z[i]
        else:
            y[i] = 0.5 * round(2.0 * z[i])

def problem_18_27(dimension, y, z):
    _problem_18_27(dimension, y, z)


cdef tuple _problem_17_26(int dimension, double[:]y):
    cdef:
        double f, f0, f1, g1
        int i, j
    f0 = 0.0
    f1 = 1.0
    g1 = 0.0
    for i in range(dimension):
        f0 += y[i]**2
    for i in range(dimension):
        f1 *= y[i] / sqrt(1.0 + i)
        g1 += _sgn(fabs(y[i]) - (f0 - y[i]**2) - 1.0)
    f = 1.0 / 4000.0 * f0 + 1.0 - f1
    return f, g1

def problem_17_26(dimension, y):
    return _problem_17_26(dimension, y)


cdef void _crossover_exp_cy(double[:, :]subpop, double[:]child, int dimension, double cr, int idx):
    cdef:
        int j_rand
    j_rand = _rand_int(0, dimension - 1)
    for j in range(dimension):
        if j_rand != j and random.random() <= cr:
            child[j] = subpop[idx, j]

def crossover_exp_cy(subpop, child, dimension, cr, idx):
    _crossover_exp_cy(subpop, child, dimension, cr, idx)


cdef void _crossover_bi_cy(double[:, :]subpop, double[:]child, int dimension, double cr, int idx):
    cdef:
        int n, count
    n = _rand_int(0, dimension - 1)
    count = 0
    while random.random() <= cr and count < dimension:
        child[(n + count) % dimension] = subpop[idx, (n + count) % dimension]
        count += 1

def crossover_bi_cy(subpop, child, dimension, cr, idx):
    _crossover_bi_cy(subpop, child, dimension, cr, idx)


cdef int _rand_choice_pb_cy(int[:]arr, double[:]pb):
    cdef:
        int selected_one
        double r
    selected_one = arr[0]
    r = random.random()
    if pb[0] <= r < pb[0] + pb[1]:
        selected_one = arr[1]
    elif pb[0] + pb[1] <= r < pb[0] + pb[1] + pb[2]:
        selected_one = arr[2]
    elif pb[0] + pb[1] + pb[2] <= r <= 1.0:
        selected_one = arr[3]
    return selected_one

def rand_choice_pb_cy(arr, pb):
    return _rand_choice_pb_cy(arr, pb)


cdef tuple _rand_choice(int size):
    cdef:
        int x_1, x_2, x_3
    x_1 = _rand_int(0, size - 1)
    x_2 = _rand_int(0, size - 1)
    x_3 = _rand_int(0, size - 1)
    while x_1 == x_2:
        x_2 = _rand_int(0, size - 1)
    while x_1 == x_3 or x_2 == x_3:
        x_3 = _rand_int(0, size - 1)
    return x_1, x_2, x_3

def rand_choice(size):
    return _rand_choice(size)

cdef void _x_correction(double[:]child, int dimension, double lb, double ub):
    for i in range(dimension):
        if child[i] < lb:
            child[i] = min(2.0 * lb - child[i], ub)
        elif child[i] > ub:
            child[i] = max(lb, 2.0 * ub - child[i])

def x_correction(child, dimension, lb, ub):
    return _x_correction(child, dimension, lb, ub)
