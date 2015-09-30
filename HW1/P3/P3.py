
# python script for problem 3
# generated from ipython notebook
import findspark
findspark.init()

from pyspark import SparkContext, SparkConf
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.ticker as ticker   


# setup spark
conf = SparkConf().setAppName('Anagram')
sc = SparkContext(conf=conf)


# In[10]:

# load textlist from net
filename = "EOWL_words.txt"

url = "https://s3.amazonaws.com/Harvard-CS205/wordlist/EOWL_words.txt"

import urllib

wordfile = urllib.URLopener()
wordfile.retrieve(url, filename)


# In[11]:

# load textfile into spark...
# each line contains one word
words = sc.textFile(filename)


# In[12]:

# map word to a pair (K, V) with K = sorted word, V = word
# for sorting use approach from http://stackoverflow.com/questions/15046242/how-to-sort-the-letters-in-a-string-alphabetically-in-python
rdd = words.map(lambda x: (''.join(sorted(x)),x))


# In[13]:

# now group words by key K (which is the sorted word)
rdd = rdd.groupByKey()


# In[14]:

# map the result to the desired result
# (sortedletters, numberofvalidanagrams, [word1, word2, ...])

# x is now a tuple
rdd = rdd.map(lambda x: (x[0], len(x[1]), list(x[1])))


# In[15]:

# sort the result after the number of valid anagrams (just for convenience)
rdd = rdd.sortBy(lambda x: x[1], ascending=False)


# In[16]:

# print out results
results = rdd.collect()


# the first entry of the results array is the wordlist, which has the most anagrams

# In[17]:

print results[0]


# In[18]:

# write first result to P3.txt,
# if everything is ok, this should return True
with open('P3.txt', 'w') as f:
    f.write(str(results[0]))
f.closed




