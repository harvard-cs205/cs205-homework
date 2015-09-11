import multiprocessing as mp
import time

def burnTime(k):
    print "Hi Job %d" %k
    time.sleep(0.25)
    print "Bye Job %d" %k
    return k

if __name__ == '__main__':
    pool = mp.Pool(4);

    result = pool.map(burnTime, range(10))
    print(result)
