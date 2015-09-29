import findspark 
findspark.init()

import pyspark
sc = pyspark.SparkContext()

wlist = sc.textFile('EOWL_words.txt')
alpha_words = wlist.map(lambda x: ("".join(sorted(x)), [x]))
alpha_anagrams = alpha_words.reduceByKey(lambda x, y: x + y)
anagram_list = alpha_anagrams.map(lambda x: (x[0], len(x[1]), x[1]))
anagram_list.collect()
max_anagram = anagram_list.sortBy(lambda x: x[1], ascending=False).take(1)
print max_anagram
