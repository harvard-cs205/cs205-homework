from P2 import *

# Your code here

# part 2b
# make pixels
i = sc.parallelize(range(2000))
j = sc.parallelize(range(2000))

pxls = i.cartesian(j)


# upt through mandelbrot func
result = pxls.map(lambda x: (x , mandelbrot(x[1]/500.0 - 2,x[0]/500.0 - 2))).partitionBy(100)


#draw_image(result)

plt.hist(sum_values_for_partitions(result).collect())
plt.savefig('P2b_hist.png')
