import numpy as np
sample_list = [0.1, 0.2, 0.3, 0.4]
main_list = [1, 2, 3, 4]
print(np.random.choice(main_list, 10, replace=True, p=sample_list))