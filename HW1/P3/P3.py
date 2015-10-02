
# coding: utf-8

# In[1]:

# github.com/minrk/findspark
import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="Spark1")
import numpy as np


# In[2]:

letters_after_a = [chr(i) for i in range(66,91)]
words = sc.textFile("./EOWL-v1.1.2/CSV_Format/"+"A"+" Words.csv")
#make one big RDD with all words
for i in range(66,91):
    letter = chr(i)
    words = words.union(sc.textFile("./EOWL-v1.1.2/CSV_Format/"+letter+" Words.csv"))


# In[29]:

#maps to K,V: words --> (sorted_letter_seq, (num_words, [word])
sorted_letter_seqs = words.map(lambda word: ("".join(sorted(list(word))), (1,[word]))).reduceByKey(lambda val1, val2: (val1[0] + val2[0],val1[1]+val2[1]))
print sorted_letter_seqs.takeOrdered(1, lambda KV: -KV[1][0])

