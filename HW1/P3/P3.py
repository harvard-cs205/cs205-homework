# Ankit Gupta
# CS 205
import pyspark
sc = pyspark.SparkContext(appName="Spark1")
import numpy as np 
import itertools


# Returns all orderings of a string
# These are not necessarily all valid words
def get_anagrams(s):
	return ["".join(perm) for perm in itertools.permutations(s)]

wlist = sc.textFile('../P3/EOWL_words.txt')

sorted_letter_to_word = wlist.map(lambda x: (''.join(sorted(x)), x))\
                            .groupByKey().map(lambda (x, y): (x, len(y), list(y)))\
                            .sortBy(lambda x: (-1)*x[1])
print sorted_letter_to_word.first()