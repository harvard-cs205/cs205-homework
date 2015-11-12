import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
sc = SparkContext()

lines = sc.textFile('EOWL_words.txt')

#(SortedLetterSequence1, NumberOfValidAnagrams1, [Word1a, Word2a, ...])
rdd1 = lines.map(lambda word: (''.join(sorted(word)), word)).groupByKey().map(lambda x : (x[0], len(list(x[1])), list(x[1])))
highest_anagram_count = rdd1.takeOrdered(1, key=lambda x: -x[1])
print highest_anagram_count