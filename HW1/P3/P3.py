

import math as ma
import findspark
import os
findspark.init() 
import pyspark
sc = pyspark.SparkContext()
import numpy as np

import string
import itertools
path = '/Users/patrickkuiper/Desktop/Academic/Harvard Courses/CS205/HW/HW1/EOWL-v1.1.2/CSV Format/'
letters = list(string.ascii_uppercase)
file_end = ' Words.csv'
word_matrix = []
for i in range(len(letters)):
    word_matrix.append(np.genfromtxt(path+letters[i]+file_end, delimiter=',', dtype = str))


word_list = itertools.chain(*word_matrix)
flat_word_list = list(word_list)


word_rdd = sc.parallelize(flat_word_list)
word_kv = word_rdd.map(lambda x: (''.join(sorted(list(x))), x)).repartition(100)
word_val = word_rdd.map(lambda x: (''.join(sorted(list(x))), 1)).repartition(100).reduceByKey(lambda a, b: a + b)


word_kv_list = word_kv.groupByKey().mapValues(list).collect()
word_kv_rdd = word_kv.groupByKey().mapValues(list)


word_join_list = word_kv_rdd.join(word_val).filter(lambda z: z[1][1] > 1).collect()


anagram_list = np.zeros(len(word_join_list))
for i in range(len(word_join_list)):
    anagram_list[i] = word_join_list[i][1][1]

for i in range(len(word_join_list)):
    if word_join_list[i][1][1] == anagram_list.max():
        longest_anagram = word_join_list[i]



print longest_anagram



