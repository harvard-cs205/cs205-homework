# calling the file with all of the imports included
execfile('P2.py')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import seaborn as sns
sns.set_context('poster', font_scale=1.25)


# initializing spark
import pyspark as ps

config = ps.SparkConf()
config = config.setMaster('local[' + str(2*mp.cpu_count()) + ']')
config = config.setAppName('P2a')

sc = ps.SparkContext(conf=config)

#creating function for creating the pixels and passing on a tuple and the results
def createPixels(row,column):
    xx = (column/500.0)-2.0
    yy = (row/500.0)-2.0
    result = mandelbrot(xx,yy)
    return (row,column),result

# this creates an RDD that constains values in the range of 0 to 2000 stepping 
# in increments of 1 each divided into 10 partitions so that 10*10=100 partitions
row = sc.parallelize(range(0,2000),10)
column = sc.parallelize(range(0,2000),10)

# this creates all possible combinations of rows and columns
joined = row.cartesian(column)


# this maps the joined RDD into the function createPixels
mandlebrot = joined.map(lambda rr: createPixels(rr[0],rr[1]))

# here I pass the RDD in the format ((I,J),V) to the pre-defined draw function
draw_image(mandlebrot)
plt.savefig('P2a_Mandelbrot')

# here I pass the same RDD in the same format to find how many times it was 
# iterated upon
sum_values=sum_values_for_partitions(mandlebrot).collect()

# this section gives me the histogram in a pretty fashion. This was done with
# the help from Bryan Weinsteing
plt.hist(sum_values, bins=np.logspace(3, 8))
plt.gca().set_xscale('log')
plt.savefig('P2a_hist.png', bbox_inches='tight')

print "the number of workers are: ", len(sum_values)
