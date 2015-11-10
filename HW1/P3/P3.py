from pyspark import SparkContext
sc = SparkContext("local", "P3")
wlist = sc.textFile("EOWL_words.txt")
sorted_wlist = wlist.map(lambda x: (''.join(sorted(x)), [x]))
reduced_wlist = sorted_wlist.reduceByKey(lambda x,y: x+y)
count_wlist = reduced_wlist.map(lambda x: (x[0], len(x[1]), x[1]))
print count_wlist.takeOrdered(1, key = lambda x: -x[1])[0]
