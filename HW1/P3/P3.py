from pyspark import SparkContext
sc = SparkContext("local", "Simple App")
words = sc.textFile("wlist.txt")

rdd = words.map(lambda x : (''.join(sorted(x)), x)) \
		   .groupByKey() \
		   .map(lambda (key,values) : (key, len(list(values)), list(values))) \
		   .sortBy(lambda (x1, x2, x3) : x2, ascending = False)

print rdd.take(10)