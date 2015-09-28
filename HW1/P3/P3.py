%matplotlib inline
import findspark
findspark.init()
import pyspark

# shut down the previous spark context
sc.stop() 
sc = pyspark.SparkContext(appName="myAppName")
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import pdb
import time
import pandas as pd
import itertools

# Load dictionary

wlist = sc.textFile('EOWL_words.txt',  use_unicode=True)

# convert into lower case words
wlist = wlist.map(lambda r: r.lower())

# sort all the words alphabetically, check if they are in the dictionary
anagram = wlist.map(lambda r: ( ''.join( sorted(r) ), r)).groupByKey().sortByKey()

# create the requested structure
myrdd = anagram.map(lambda r: (r[0], len(r[1]), r[1] ))

# Collect
p = myrdd.collect()

# Identify the words that have the maximum anagrams
tom = [s[1] for s in p]
m = max(tom)
l_max = [i for i,j in enumerate(tom) if j == m ]

# Get the words
myanagrams = [p[l] for l in l_max]
myanagrams


# Copy them and their anagrams in a text file
p3 = open('P3.txt', 'w')

for i in range(0,3):
    p3.write('word: ')
    p3.write( myanagrams[i][0]+' --> ')
    p3.write('anagrams : ')
    for words in myanagrams[i][2]:
        p3.write(words+ ' , ')
    p3.write('\n\n')



p3.close()