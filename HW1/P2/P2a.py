from P2 import *

# Your code here


# make pixels
i = sc.parallelize(range(1000))
j = sc.parallelize(range(1000))

pxls = i.cartesian(j)



# put through mandelbrot func
result = pxls.map(lambda x: (x , mandelbrot(x[0]/500.0 - 2,x[1]/500.0 - 2)))



#draw_image(result)
plt.hist(sum_values_for_partitions(result).collect())
plt.savefig('P2a_hist2.png')

print "got here"
