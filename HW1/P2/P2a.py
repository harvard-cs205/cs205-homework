import findspark
findspark.init()

from P2 import *
from pyspark import SparkContext, SparkConf

# setup spark
conf = SparkConf().setAppName('Mandelbrot')
sc = SparkContext(conf=conf)

# create the space
nx, ny = (2000, 2000)
x = np.linspace(-2, 2, nx)
y = np.linspace(-2, 2, ny)

xv, yv = np.meshgrid(x, y)

# compute coordinate list
coords = zip(list(xv.ravel()), list(yv.ravel()))

# create rdd with 100 partitions
num_partitions = 100
rdd = sc.parallelize(coords, num_partitions)

# now map using mandelbrot
mb = lambda c: mandelbrot(c[0], c[1])

rdd = rdd.map(mb)

# collect data & print image
data = rdd.collect()

# plotting
im = np.array(data).reshape((nx, ny))
im = np.log(im + 1) # log intensity makes it easier to see levels
plt.imshow(im, cmap=cm.gray)
plt.savefig('mandelbrot.png')
plt.show()