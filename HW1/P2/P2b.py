from P2 import *

# initialize spark context
sc = pyspark.SparkContext("local[4]", "Spark1")

# create RDD of pixel location tuples
i_vals = sc.parallelize(range(2000))
ij_vals = i_vals.cartesian(i_vals)

# associate mandelbrot values with each pixel
ijxy_vals = ij_vals.zip(ij_vals).mapValues(lambda (i,j): (j/500 - 2, i/500 - 2))
mandel_vals = ijxy_vals.partitionBy(100).mapValues(lambda xy: mandelbrot(*xy))
#draw_image(mandel_vals)
#draw_partitions(mandel_vals)

plt.hist(sum_values_for_partitions(mandel_vals).collect(), bins=200, range=(0,2.5e7))
plt.xlabel('iteration steps')
plt.ylabel('partitions')
plt.savefig('P2b_hist.png')
