import numpy as np

a=np.load('ADP_COST.npy')
Cost=a.tolist()

import matplotlib.pyplot as plt

plt.hist(Cost[400:500],bins=25)
plt.show()