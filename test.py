
import numpy as np

x = np.random.random((20, 3))
y = x.tolist()
while x.shape[0] > 2:
        x = np.delete(x, x.shape[0] - 1, 0)
while len(y) > 10:
        y.pop(len(y) - 1)
z = 0
