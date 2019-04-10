import pyspark
import numpy as np
from pyspark import SparkContext
sc = SparkContext()

#read data and sort them
words = sc.textFile('EOWL_words.txt').map(lambda x: ("".join(np.sort(list(x))),x)).groupByKey().mapValues(list)
rdd_res = words.map(lambda x: (x[0],(len(x[1]),x[1])))

#find the ones have 11 anagrams
res = rdd_res.filter(lambda x: x[1][0] == 11).collect()

#write the txt file
f = open('P3.txt', "w")
for i in res: f.write(str(i)+"\n")
f.close()
