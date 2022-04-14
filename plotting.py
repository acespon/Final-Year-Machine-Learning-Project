import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 6)
y = np.arange(1, 6)


def recurse(x):
    if x == 1:
        return 3
    else:
        return ((2 * x) - 1) * ((2 * x) + 1) + recurse(x - 1)


for i in range(5):
    y[i] = recurse(x[i])
    print(recurse(x[i]))

plt.plot(x, y, 'bo')

plt.show()
