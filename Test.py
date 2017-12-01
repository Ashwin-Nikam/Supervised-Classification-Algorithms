import numpy as np
a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
b = np.random.choice(a, len(a), replace=True)
print(b)