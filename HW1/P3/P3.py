import pyspark

sc = pyspark.SparkContext()

# read in file
wlist = sc.textFile('EOWL_words.txt')

# an RDD of tuples (word_ascending, [word])
wlist_sorted = wlist.map(lambda x: (''.join(sorted(x)), [x]))

# combine words with the same sequence and get rid of those without anagram
wlist_reduced = wlist_sorted.reduceByKey(lambda x, y: x+y).filter(lambda x: len(x[1])>1)

# create an RDD of the format asked in the problem
anagram = wlist_reduced.map(lambda x: (x[0], len(x[1]), x[1]))

# print top five anagrams
print anagram.takeOrdered(5, key=lambda x: -x[1])