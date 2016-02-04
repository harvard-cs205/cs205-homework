import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	x = np.arange(1.0, 128.0+1, 1.0)
	y1 = np.ceil(np.log2(x))
	y2 = x-1
	plt.plot(x, y1, 'r-', label='Parallel')
	plt.plot(x, y2, 'b-', label='Solo')
	plt.legend(loc='upper left')
	plt.xlim(1.0, 128.0)
	plt.xlabel('Number of Bags')
	plt.ylabel('Time (seconds)')
	plt.savefig('P5.png')