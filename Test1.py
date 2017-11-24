import numpy as np

column = ["present", "absent", "present", "hahaha","banana","banana"]
d = dict([(y,x) for x,y in enumerate(sorted(set(column)))])
print(d.keys())
print(d.values())