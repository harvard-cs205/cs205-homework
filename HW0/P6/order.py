import numpy as np
import multiprocessing as mp

def main():
	N = 3
	A = np.random.uniform(-1,1,(10,N,N))
	print A

	res = np.eye(N)
	print res
	for i in xrange(10):
		mult(A[i])
	print res

	pool = mp.Pool(4)
	res = np.eye(N)
	pool.map(mult, A)
	print res

# Performs A = A * B
def mult(A):
	res = np.dot(res,A)

if __name__ == '__main__':
	main()
