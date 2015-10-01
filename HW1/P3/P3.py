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

start = time.time()

# Load dictionary

wlist = sc.textFile('EOWL_words.txt',  use_unicode=True)

# convert into lower case words
wlist = wlist.map(lambda r: r.lower())

# sort all the words alphabetically, check if they are in the dictionary
anagram = wlist.map(lambda r: ( ''.join( sorted(r) ), r)).groupByKey().sortByKey()

# create the requested structure
myrdd = anagram.map(lambda r: (r[0], len(r[1]), r[1] ))

# Get the largest valid number of anagrams
myanagrams = myrdd.takeOrdered(1 ,key=lambda x: -x[1] )

# Copy them and their anagrams in a text file
p3 = open('P3.txt', 'w')


p3.write('word: ')
p3.write( myanagrams[0][0]+' --> ')
p3.write('anagrams : ')
for words in myanagrams[0][2]:
    p3.write(words+ ' , ')
p3.write('\n\n')
p3.close()
stop = time.time()
elapse = stop - start
print elapse