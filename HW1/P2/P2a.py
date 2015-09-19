from P2 import *
import pyspark

# Your code here
TOTAL_PIXELS = 2000

def calcBase(i):
    return i / 500.0 - 2

def calcX(ij):
    return calcBase(ij[1])

def calcY(ij):
    return calcBase(ij[0])

if __name__ == '__main__':
    imageIJ = []
    for i in xrange(TOTAL_PIXELS):
        for j in xrange(TOTAL_PIXELS):
            imageIJ.append((i, j))

    sc = pyspark.SparkContext(appName='YK-P2a')
    data = sc.parallelize(imageIJ, 100)
    rdd = data.map(lambda ij: (ij, mandelbrot(calcX(ij), calcY(ij))))
    
    draw_image(rdd)
