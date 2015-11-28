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