import multiprocessing as mp
import time

def burnTime(k):
    print("Hi Job {}".format(k))
    time.sleep(0.25)
    print("Bye Job {}".format(k))
    return k

if __name__ == '__main__':
    pool = mp.Pool(4)  # Create a pool of 4 processes

    # Apply burnTime to this list of "job numbers" using the pool
    result = pool.map(burnTime, range(10))
    print(result)




import numpy as np

#A = np.matrix([[3, 5, 9], [2, 7, 4]])

A = np.random.rand(5, 3)

B = np.dot(A, A.T)
C = np.dot(A.T, A)


print np.linalg.eig(B)
print np.linalg.eig(C)