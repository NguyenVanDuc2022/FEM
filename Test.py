import numpy as np

t = np.array([[2, 3, 0, 5, 6, 1, 7, 8, -9, 0]])
t1 = max(t[0,:], key=abs)
print(t.max())
print(t1)

