# File used with Spark on Shell
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Loading the data
wlist = sc.textFile('EOWL_words.txt')

# sorting the anagram
word = wlist.map(lambda x: (''.join(sorted(x)), ''.join(x)))

anagram = word.groupByKey().map(lambda x: (x[0], len(list(x[1])), list(x[1])))

top = anagram.takeOrdered(1, key=lambda x: -x[1])

print top
