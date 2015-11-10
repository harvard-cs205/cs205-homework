from P2 import *
import itertools
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

import findspark
findspark.init()
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName('P2').setMaster('local')
sc = SparkContext(conf=conf)

# Calculate a mandelbrot set in parallel manner (with load balancing)

data = list(itertools.permutations(range(1, 2001), 2))
shuffle(data)
distData = sc.parallelize(data, 100)
point_intensity = distData.map(lambda (x, y): ((x, y), mandelbrot((x/500.0) - 2, (y/500.0) - 2))).cache()

draw_image(point_intensity)

partition_load = sum_values_for_partitions(point_intensity).collect()

plt.hist(partition_load)

plt.title('Distribution of computing load between partitions')  
plt.xlabel('Computing load') 
plt.ylabel('Number of partitions')
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.30)
plt.savefig('P2b_hist.png')
