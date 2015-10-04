from pyspark import SparkContext
sc=SparkContext()

rdd=sc.textFile("/home/memory/cs205-homework/HW1/P3/EOWL_words.txt")

final=rdd.flatMap(lambda x: x.split()).map(lambda x: (''.join(sorted(list(x))),x)).groupByKey().mapValues(list).map(lambda x: (x[0],len(x[1]),x[1]))

print final.takeOrdered(1, lambda x: -x[1])
