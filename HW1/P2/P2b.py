# P2 solution part b
from P2 import *
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext

def IJtoXY(I, J, pixel):
    return J * 4.0 / pixel - 2, I * 4.0 / pixel - 2 

if __name__ == '__main__':
    
    conf = SparkConf().setAppName('KaiSquare')
    sc = SparkContext(conf = conf)

    pixel = 2000
    
    I = sc.parallelize(range(pixel + 1))
    J = sc.parallelize(range(pixel + 1))

    IJ = I.cartesian(J)
    IJ = IJ.partitionBy(100) # this is different from part a
    
    IJK = IJ.map(lambda ij: (ij, mandelbrot(* IJtoXY(ij[0], ij[1], pixel))))
    numpart = sum_values_for_partitions(IJK)
    plt.hist(np.log10(np.array(numpart.collect()) + 1), bins=np.linspace(0, 8, 17))
    plt.xlabel('log10(Number) of Works')
    plt.ylabel('Number of Partitions')
    plt.title('Works per Partition')
    plt.show()

    draw_image(IJK)
