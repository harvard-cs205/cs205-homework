
# coding: utf-8

# # Problem 6

# In[1]:

import findspark
findspark.init()

from pyspark import SparkContext, SparkConf
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.ticker as ticker   


# setup spark
conf = SparkConf().setAppName('MarkovShakespeare')
sc = SparkContext(conf=conf)


# In[2]:

# load textlist from net
filename = "pg100.txt"

url = "http://www.gutenberg.org/cache/epub/100/pg100.txt"

import urllib

wordfile = urllib.URLopener()
wordfile.retrieve(url, filename)


# In[3]:

# the texts from shakespeare are contained in lines 182 - 124367
# create the file Shakespeare.txt using the line range from above
# if all is ok, this should return True, True
with open(filename, 'r') as f:
    lines = f.readlines()
    lines = lines[181:124367]
    with open('Shakespeare.txt', 'w') as f2:
        f2.writelines(lines)
    print f2.closed
f.closed


# In[51]:

# filtering is done by spark
words = sc.textFile('Shakespeare.txt')


# In[52]:

# split lines into words (after whitespace)
# before splitting replace first some escape characters with white space
rdd = words.flatMap(lambda line: line.replace('\n', ' ').replace('\r', ' ').split(' '))


# In[53]:

# now perform filtering on the words
# to develop the regular expression this awesome tool was used https://regex101.com

# (1) filter out words that only contain numbers
# (2) filter out words for which all letters are capitalized
# (3) filter out words that contain letters only and end with a period
import re

pattern1 = re.compile(ur'([0-9]+\.?)|\.')  # use only a simple RE for numbers here (numbers in the shakespeare text are formateed as 1., 2., ...)
pattern2 = re.compile(ur'([A-Z]+\.?)|\.') # ==> do (2) & (3) together in one regex!

# filter does not change the order
rdd = rdd.filter(lambda x: not pattern1.match(x)).filter(lambda x: not pattern2.match(x)).filter(lambda x: len(x) > 0)


# In[54]:

#rdd.collect()


# In[55]:

# zip with index
rdd = rdd.zipWithIndex()


# In[56]:

#rdd.take(5)


# In[57]:

# map to list
rdd = rdd.flatMap(lambda x: [(x[1], (0, x[0])), (x[1] - 1, (1, x[0])), (x[1] - 2, (2, x[0]))])


# In[58]:

#rdd.take(5)


# In[59]:

rdd = rdd.groupByKey()


# In[60]:

#rdd.take(5)


# In[61]:

# map (keyA, (keyA, wordA), (keyA + 1, wordA+1), (keyA + 2, wordA+2)) to (wordA, wordA+1, wordA+2)
fun = lambda x: list(x[1])
rdd = rdd.map(fun)


# # In[62]:

# # map such that the three words form the key with value 1, we use that then to sum up everything
# #gfun = lambda x: ((x[0][1], x[1][1], x[2][1]), 1)

# # try to combine
# gfun = lambda x: (x[0][1] + x[1][1] + x[2][1], 1)
# #gfun = lambda x: (x[0][1], 1)
# #rdd = rdd.map(gfun)


# # In[63]:

# temp = rdd.map(gfun)


# # In[64]:

# #temp.take(20)


# # In[66]:

# temp.reduceByKey(lambda x,y:x+y).collect() 
# #temp.take(5)


# # In[ ]:



