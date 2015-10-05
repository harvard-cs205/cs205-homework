
# coding: utf-8

# In[1]:

import findspark
import os
findspark.init('/home/chongmo/spark') # you need that before import pyspark.
import pyspark
from pyspark import SparkContext
sc =SparkContext()


# In[2]:

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm


# In[3]:

wlist = sc.textFile('EOWL_words.txt')


# In[7]:

#convert to a new RDD in the form: (SortedLetterSequence, [Word1a, Word2a, ...])
newlist=wlist.keyBy(lambda x: ''.join(sorted(x))).groupByKey().sortByKey() 


# In[17]:

Jumble_RDD=newlist.map(lambda x: (x[0], len(list(x[1])), list(x[1])))


# In[19]:

max(Jumble_RDD.collect(), key=lambda x: x[1])




