"""
Written by Jaemin Cheun!
Harvard CS205
Assignment 1
October 6, 2015
"""

import numpy as np
import findspark
findspark.init()

from pyspark import SparkContext

# initiaize Spark
sc = SparkContext("local", appName="P3")
sc.setLogLevel("ERROR")
lines = sc.textFile("EOWL_words.txt")

# first make a tuple where key : the word (sorted) and value : the word
sorted_list = lines.map(lambda word : (''.join(sorted(list(word))), [word]))

# we then reducebykey so that the anagrams are collected together
anagrams = sorted_list.reduceByKey(lambda x,y : x + y)

# we then add the Number of Valid Anagrams count
anagram_list = anagrams.map(lambda x : (x[0], len(x[1]), x[1]))
print anagram_list.takeOrdered(1, key = lambda x: -x[1])