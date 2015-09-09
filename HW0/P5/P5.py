import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
if __name__ == 'main':
	ax = Axes3D(plt.figure())
	N = [2**i for i in range(8,100)]
	dt = 1
	time_take = []
	for i in N:
		j = 0
		while True:
			if i/2 != 0:
				i = i/2
				j+=1
				continue
			time_take.append(j)
			break
	print time_take
	ax.plot(N,time_take,'.g', label='aha')
	ax.legend()
	plt.show()


