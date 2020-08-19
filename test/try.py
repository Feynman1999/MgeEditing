import megengine
import numpy as np

a = megengine.tensor(np.zeros((3,3,3,3), dtype = np.float32))

print(a[:, 0, ...].shape)