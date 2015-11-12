######################################################
### Problem 3 - Anagram Solver [10%]			   ###
### P3.py									       ###
### Patrick Day 								   ###
### CS 205 HW1                                     ###
### Oct 4th, 2015								   ###
######################################################

########################
### Import Functions ###
########################
import pyspark
import numpy as np

import os
sc = pyspark.SparkContext()
sc.setLogLevel('ERROR')

### Import Data and Format without Unicode
wlist = sc.textFile('EOWL_words.txt', use_unicode=False)

# Create RDD of sorted letter sequences and original word
sort_let_seq = wlist.map(lambda x: (''.join(sorted(list(x))), x))

# Referenced from http://spark.apache.org/docs/latest/api/python/pyspark.html
# Create RDD of sorted letters k, and combinations of words v
SLS_word = sort_let_seq.groupByKey().mapValues(list)

# Create new RDD with Sorted Sequence, Respective len, and list of words
final_rdd = SLS_word.map(lambda (sort, words): (sort, len(words), words))

# Sort and take the first entry
most_ana = final_rdd.sortBy(lambda x: x[1], ascending=False).first()
print(most_ana)

