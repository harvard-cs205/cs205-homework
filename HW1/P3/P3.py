from os import walk
import findspark
findspark.init('/home/toby/spark')

import pyspark
import numpy as np

sc = pyspark.SparkContext(appName="Spark 2")
myfiles = "./words/*.csv"
words = sc.textFile(myfiles)

words = words.map(lambda word: ["".join(sorted((word))), [word]])
words = words.reduceByKey(lambda a, b: a+b)
print words.take(10)[0]

words = words.map(lambda element: [element[0], len(element[1]), element[1]])
#print words.take(10)
print words.takeOrdered(1, key=lambda x: -x[1])
