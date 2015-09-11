import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
  bags = range(1,100)
  ptime = [np.ceil(np.log(b)) for b in bags]
  ntime = [b - 1 for b in bags]
  plt.plot(bags, ntime)
  plt.plot(bags, ntime)