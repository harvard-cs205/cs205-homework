import numpy as np
import matplotlib.pyplot as plt
import math

# Main
if __name__ == '__main__':
	x = np.linspace(1, 1000, 20)
	y1 = [math.log(i, 2) for i in x]
	y2 = [(i - 1) for i in x]
	plt.plot(x, y1, '-b', label = 'Infinite number of employees')
	plt.plot(x, y2, '--r', label = 'Lone cashier')
	plt.xlabel('Bag count')
	plt.ylabel('Counting time')
	plt.title('Counting Time for Bags: Parallel vs Serial')
	plt.legend(loc = 2)
	plt.show()