import findspark
findspark.init()
import os
import pyspark
sc = pyspark.SparkContext()

wlist = sc.textFile('EOWL_words.txt')
wlistrdd = wlist.map(lambda x: ["".join(sorted(x)), [x]])
wlistrdd1 = wlistrdd.reduceByKey(lambda x,y: x + y)
wlistrdd2 = wlistrdd1.map(lambda x: (x[0], len(x[1]), x[1]))

maxlength = wlistrdd2.takeOrdered(20, key = lambda x: -x[1])[0][1]
topanagram = wlistrdd2.filter(lambda x: x[1] == maxlength)
print topanagram.collect()