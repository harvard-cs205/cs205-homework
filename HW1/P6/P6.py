import pyspark
from pyspark import SparkContext
import numpy as np
sc = SparkContext()
# initial number of partitions 
num_partition = 20
# Read in files
pg100 = sc.textFile("pg100.txt").collect()

rdd = sc.parallelize(pg100, num_partition).flatMap(lambda x: x.split())

# filter out words
def contextFilter(x):
	if x.isdigit():
		return False
	elif x.isupper():
		return False
	elif x[:-1].isupper():
		return False
	return True

# create tuple list
def tuplelist(rdd):
	words1 = rdd.collect()
	words2 = words1[1:] + [' ']
	words3 = words2[1:] + [' ']

	def parallel(word):
		return sc.parallelize(word, num_partition).zipWithIndex().map(lambda x: (x[1], x[0]))

	rdd = parallel(words1).union(parallel(words2)).union(parallel(words3))
	return rdd

# create format of ((word1, word2), word3)
def createTuple(x):
	num = tuple(x[1])
	return ((num[0], num[1]), num[2])

rdd = rdd.filter(contextFilter)
rdd = tuplelist(rdd)
rdd = rdd.groupByKey().map(createTuple)
rdd = sc.parallelize(rdd.countByValue().items()).map(lambda x: (x[0][0], [(x[0][1], x[1])]))
rdd = rdd.combineByKey(lambda x: x, lambda x, y: x+y, lambda x,y: x+y)
diction = rdd.collectAsMap()

# generate words.
phrases = sc.parallelize(rdd.takeSample(False, 10, 256))
phrases = phrases.map(lambda x: [x[0][0], x[0][1]])

# get the third word
def obtainWord(x):
    words = []
    dictionary = diction[x]
    for key, value in dictionary:
    	for i in xrange(value):
    		words.append(key)
    return np.random.choice(words)

# iterate to obtain words
for i in xrange(18):
    phrases = phrases.map(lambda x: x + [obtainWord((x[-2], x[-1]))])

phrases = phrases.map(lambda x: " ".join(x)).collect()

# write the output into files
with open("P6.txt", "w") as f:
	for i in phrases:
		f.write(i)
		#f.write('Phrase ' + str(i + 1) + '\n')
		f.write("\n\n")


