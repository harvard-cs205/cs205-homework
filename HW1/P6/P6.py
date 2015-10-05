from pyspark import SparkContext
import numpy as np
sc =SparkContext()

nPartitions = 20
# Read in files
data = sc.textFile("shakespear.txt").collect()

rdd = sc.parallelize(data, nPartitions).flatMap(lambda x: x.split())
# Function that filters out words as required.
def contextFilter(rdd):
	rdd = rdd.filter(lambda x: x.isdigit() == False)
	rdd = rdd.filter(lambda x: x.isupper() == False)
	rdd = rdd.filter(lambda x: x[:-1].isupper() == False)
	return rdd

rdd = contextFilter(rdd)

# For each word, add its first and second lag.
def concatenate(rdd):
	words1 = rdd.collect()
	words2 = words1[1:] + ["/"]
	words3 = words2[1:] + ["/"]

	rdd1 = sc.parallelize(words1, nPartitions).zipWithIndex().map(lambda x: (x[1], x[0]))
	rdd2 = sc.parallelize(words2, nPartitions).zipWithIndex().map(lambda x: (x[1], x[0]))
	rdd3 = sc.parallelize(words3, nPartitions).zipWithIndex().map(lambda x: (x[1], x[0]))
	rdd = rdd1.union(rdd2).union(rdd3)
	return rdd

rdd = concatenate(rdd)

# Place the word into the format of ((word1, word2), word3)
def placeTuple(x):
	elem = tuple(x[1])
	return ((elem[0], elem[1]), elem[2])

rdd = rdd.groupByKey().map(placeTuple)
rdd = sc.parallelize(rdd.countByValue().items()).map(lambda x: (x[0][0], [(x[0][1], x[1])]))
rdd = rdd.combineByKey(lambda x: x, lambda x, y: x+y, lambda x,y: x+y)
dictionary = rdd.collectAsMap()

# Start generating words.
phrases = sc.parallelize(rdd.takeSample(False, 10, 225))
phrases = phrases.map(lambda x: [x[0][0], x[0][1]])

# Get the third word of the phrases iteratively
def getWord(x):
    third = dictionary[x]
    words = []
    for (k,v) in third:
    	words += [k for i in xrange(v)]
    return np.random.choice(words)

# iterate to obtain 20 words total
for i in xrange(18):
    phrases = phrases.map(lambda x: x + [getWord((x[-2],x[-1]))])

# Convert list of lists to list of strings and output phrases.
phrases = phrases.map(lambda x: " ".join(x)).collect()

with open("P6.txt", "w") as txtfile:
	for p in phrases:
		txtfile.write(p)
		txtfile.write("\n")
		txtfile.write("\n")


