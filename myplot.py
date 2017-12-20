import matplotlib.pyplot as plt
import numpy as np

def myplot(arr):
    plt.imshow(arr, interpolation=None)
    plt.savefig('class.png', dpi=1000)
    plt.show()

dat = np.load('class_pmi.npy')
myplot(dat)

# 60, 103
# 7, 140
# ...