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

def createPixels(row,column):
    xx = (column/500.0)-2.0
    yy = (row/500.0)-2.0
    result = mandelbrot(xx,yy)
    return (row,column),result

row = sc.range(0,2000)
column = sc.range(0,2000)

joined = row.cartesian(column)
mandlebrot = joined.map(lambda rr: createPixels(rr[0],rr[1]))
draw_image(mandlebrot)
plt.show()

sum_values=sum_values_for_partitions(mandlebrot).collect()

plt.hist(sum_values, bins=np.logspace(3, 8))
plt.gca().set_xscale('log')
plt.savefig('P2a_hist.png', bbox_inches='tight')
