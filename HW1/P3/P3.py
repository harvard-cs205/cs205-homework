import numpy as np
import matplotlib.pyplot as plt

import findspark
findspark.init()
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName('P2').setMaster('local')
sc = SparkContext(conf=conf)

# Find combination of letters that can produce most anagrams

def anagram(x):
	wlist = sc.textFile('./words.txt')
	rdd1 = wlist.map(lambda x: (''.join(sorted(x)), x))
	rdd2 = rdd1.groupByKey().map(lambda x : (x[0], len(x[1]), list(x[1])))
	return rdd2.takeOrdered(x, key=lambda x: -x[1])

print anagram(4)