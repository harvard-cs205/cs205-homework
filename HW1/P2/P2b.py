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
    
    xisrdd = sc.parallelize(xrange(0,2000)) #no partition    
    cartrdd = xisrdd.cartesian(xisrdd) #generate coordinate
    
    balance = cartrdd.partitionBy(numPart)
    cordrdd = balance.map(pixel2cord) #compute part
    #print cordrdd.glom().collect()
    
    effort = sum_values_for_partitions(cordrdd)
    #print effort.collect()
    plt.hist(effort.collect())#, bins=numPart)
    plt.savefig("P2b_hist.png")
    
    draw_image(cordrdd)
