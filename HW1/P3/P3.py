
# coding: utf-8

# In[1]:

# github.com/minrk/findspark
import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="Spark1")


# In[17]:

# Load the word list
word_list = sc.textFile('../../EOWL-v1.1.2/CSV Format/*.csv')
word_list.count()


# In[33]:

# Sort the letters of each word
word_list_keyed = word_list.map(lambda x: (''.join(sorted(x)), x))


# In[54]:

# Extract the # of anagrams for each unique letter combination
word_list_grped = word_list_keyed.groupByKey().map(
    lambda x: (x[0], len(x[1]), list(x[1])))


# In[62]:

# Find the letter combination with the max # of anagrams
max_anagram = word_list_grped.takeOrdered(1, key=lambda x: -x[1])
with open("P3.txt", "w") as text_file:
    text_file.write("{}".format(max_anagram))

