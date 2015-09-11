import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
  bags = range(1,32)
  ptime = [np.ceil(np.log(b)) for b in bags]
  ntime = [b - 1 for b in bags]
  plt.scatter(bags, ntime, c="red")
  plt.scatter(bags, ptime, c="green")
  plt.xlabel('Number of bags to add')
  plt.ylabel('Time')
  plt.title('Parallel (green) vs Sequential (red) Addition')

  plt.show()