
# coding: utf-8

# In[1]:

import findspark
import os
findspark.init('/home/chongmo/spark') # you need that before import pyspark.
import pyspark
from pyspark import SparkContext
sc =SparkContext()


# In[3]:

S_RDD= sc.textFile('100.txt')


# In[6]:

#filter out and give words index
W_RDD=S_RDD.flatMap(lambda x: x.split()).filter(lambda x: (not x.isdigit()) & (not x.isupper()) & (not x.replace('.','').isupper())).zipWithIndex().cache()


# In[9]:

# use index/3 and index%3 to keep the order of all three connected words
# Example: The Project Gutenberg EBook of The Complete Works of William Shakespeare, by William Shakespeare
# W1_RDD gives all three words: ((u'The', u'Project'), u'Gutenberg'), ((u'EBook', u'of'), u'The'), ...
W1_RDD=W_RDD.map(lambda x: (x[1]/3, (x[0], x[1]%3))).groupByKey().filter(lambda x: len(x[1])==3).map(lambda x: ((list(x[1])[0][0], list(x[1])[1][0]), list(x[1])[2][0]))


# In[13]:

# W2_RDD gives all three words: ((u'Project', u'Gutenberg'), u'EBook'), ((u'of', u'The'), u'Complete'), ...
W2_RDD=W_RDD.map(lambda x: (x[1]/3+1 if x[1]%3!=0 else x[1]/3, [x[0], x[1]%3])).groupByKey().filter(lambda x: len(x[1])==3).map(lambda x: ((list(x[1])[0][0], list(x[1])[1][0]), list(x[1])[2][0]))


# In[16]:

# W3_RDD gives all three words: ((u'Gutenberg', u'EBook'),u'of'), ((u'The', u'Complete'), u'Works'), ...
W3_RDD=W_RDD.map(lambda x: (x[1]/3+1 if x[1]%3==2 else x[1]/3, [x[0], x[1]%3])).groupByKey().filter(lambda x: len(x[1])==3).map(lambda x: ((list(x[1])[0][0], list(x[1])[1][0]), list(x[1])[2][0]))


# In[18]:

#combine the three RDD
Triple_W_RDD=W1_RDD.union(W2_RDD).union(W3_RDD)


# In[19]:

from collections import Counter
Markov_RDD=Triple_W_RDD.reduceByKey(lambda x, y: x + ' '+y).mapValues(lambda x: Counter(x.split()).items())


# In[32]:

def random_phrases(RDD, length):
    phrase=[]  
    triple=Markov_RDD.takeSample(False, 1)
    word1=triple[0][0][0]
    word2=triple[0][0][1]
    word3=max(triple[0][1],key=lambda item:item[1])[0]
    phrase.extend([word1, word2, word3])
    while len(phrase)<length:  
        new_pair=(word2, word3)
        word1=word2
        word2=word3
        new_value=Markov_RDD.map(lambda x: x).lookup(new_pair)[0]
        word3=max(new_value,key=lambda item:item[1])[0]
        phrase.append(word3)   
    return phrase


# In[33]:

random_phrases(Markov_RDD, 20)




