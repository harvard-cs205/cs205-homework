import findspark
findspark.init('/Users/george/Documents/spark-1.5.0')
from pyspark import SparkContext

sc = SparkContext()

# load data
words = sc.textFile('EOWL_words.txt')

# make sorted element the key
sorted_key = words.map(lambda x: (''.join(sorted(x)), [x]))

# merge lists
reduced = sorted_key.reduceByKey(lambda x,y: x+y)

# create desired pset format
answer = reduced.map(lambda (x, y): (x, len(y), y))

# take the element with most anagrams
print answer.takeOrdered(1, key=lambda x: -x[1])[0]
