from P2 import *

# Your code here
from pyspark import SparkContext

def pixel2cord(p):
    i,j = p
    x = j/500.0 - 2
    y = i/500.0 - 2
    itr = mandelbrot(x,y)
    return p,itr

if __name__ == '__main__':
    sc = SparkContext("local", appName="Spark1")
    
    sidePart = 10
    numPart = sidePart*sidePart
    
    xisrdd = sc.parallelize(xrange(0,2000), sidePart) #default partition will be 10x10=100
    
    cartrdd = xisrdd.cartesian(xisrdd) #generate coordinate
    cordrdd = cartrdd.map(pixel2cord) #compute part
    #print cordrdd.glom().collect()
    
    effort = sum_values_for_partitions(cordrdd)
    #print effort.collect()
    plt.hist(effort.collect())#, bins=numPart)
    plt.savefig("P2a_hist.png")
    
    draw_image(cordrdd)
    
