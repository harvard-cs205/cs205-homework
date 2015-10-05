
# (c) L.Spiegelberg 2015
# python script for problem 3
import findspark
findspark.init()

import urllib
from pyspark import SparkContext, SparkConf

# setup spark
conf = SparkConf().setAppName('Anagram')
sc = SparkContext(conf=conf)

# load textlist from net
filename = "EOWL_words.txt"

url = "https://s3.amazonaws.com/Harvard-CS205/wordlist/EOWL_words.txt"

wordfile = urllib.URLopener()
wordfile.retrieve(url, filename)

# load textfile into spark...
# each line contains one word
words = sc.textFile(filename)


# map word to a pair (K, V) with K = sorted word, V = word
# for sorting use approach from http://stackoverflow.com/questions/15046242/how-to-sort-the-letters-in-a-string-alphabetically-in-python
rdd = words.map(lambda x: (''.join(sorted(x)),x))

# now group words by key K (which is the sorted word)
rdd = rdd.groupByKey()

# map the result to the desired result
# (sortedletters, numberofvalidanagrams, [word1, word2, ...])
# if a word is no anagram (i.e. not other word exists that under some permutation is
# equal to the given one) the numverofvalidanagrams is still 1

# x is now a tuple
rdd = rdd.map(lambda x: (x[0], len(x[1]), list(x[1])))

# sort the result after the number of valid anagrams (just for convenience)
rdd = rdd.sortBy(lambda x: x[1], ascending=False)

# print out results
results = rdd.collect()

# the first entry of the results array is the wordlist, which has the most anagrams
# write first result to P3.txt,
# if everything is ok, f.closed should return True
with open('P3.txt', 'w') as f:
    f.write(str(results[0]))
f.closed




