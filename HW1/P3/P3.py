
import pyspark
from pyspark import SparkContext
sc = SparkContext()

# read in data
# wlist = sc.textFile('s3://Harvard-CS205/wordlist/EOWL_words.txt')
wlist = sc.textFile('words.txt')

# make keys for later groupByKey()
rdd = wlist.map(lambda r: (''.join(sorted(list(r))), r))
rdd2 = rdd.groupByKey().mapValues(list)

# create the desired rdd
rdd_w = rdd2.map(lambda r: (r[0], len(r[1]), r[1]))

# print rdd_w.collect()

print rdd_w.takeOrdered(1, lambda x: -x[1])

