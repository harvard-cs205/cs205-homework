# Your code here
import pyspark
sc = pyspark.SparkContext(appName="Spark1")

# make spark shut the hell up
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

import numpy as np 
import itertools

accum = sc.accumulator(0)

linklist = sc.textFile('links-simple-sorted.txt')
split_list = linklist.map(lambda x: x.split(':'))
print split_list.take(10)

edges = split_list.flatMapValues(lambda x: x)

print edges.take(10)