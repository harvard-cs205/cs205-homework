
# coding: utf-8

# In[1]:

import findspark
findspark.init()
from collections import Counter
import pyspark
sc = pyspark.SparkContext(appName="Spark1")
import numpy as np


# In[2]:

full_text = sc.textFile("./pg100.txt")
#clean data
cleaner = lambda word: False if (word.isupper() or word.isdigit() or word == u'' or (word[:-1].isupper() and word[-1] == u'.')) else True
full_text = full_text.flatMap(lambda line: line.split(u" ")).filter(cleaner)

#get list of words in order
word_list = full_text.collect()
#get list of all words that can be first word (of 3)
first_words = sc.parallelize(word_list[:-2]) #remove last two words
#get list of all words that can be second word (of 3)
second_words = sc.parallelize(word_list[1:-1]) #remove first word
#get list of all words that can be third word (of 3)
third_words = sc.parallelize(word_list[2:]) #remove first two words
#third_words = third_words.map(lambda word: [(word,1)]) #add count
third_words = third_words.map(lambda word: Counter({word:1})) 
word_lists = [first_words, second_words, third_words]
#zip with index: word => (word, idx), then swap the key and value so we get (idx, word)
indexed_word_lists = [lst.zipWithIndex().map(lambda (word, idx): (idx, word)).partitionBy(100) for lst in word_lists]
#check copartitioned
assert indexed_word_lists[0].partitioner == indexed_word_lists[1].partitioner 
assert indexed_word_lists[1].partitioner == indexed_word_lists[2].partitioner 

# get RDD of the form ((first_word, second_word), [(third_word, count_of_third_word)])
ugly_format_first_two_words = indexed_word_lists[0].join(indexed_word_lists[1]).join(indexed_word_lists[2])
nice_rdd = ugly_format_first_two_words.map(lambda (idx,val): (val[0],val[1]))


# In[3]:

#for each (first,second) key get list of (third_word, count) tuples
nice_rdd = nice_rdd.reduceByKey(lambda a, b: a + b).mapValues(lambda val: val.items())


# In[4]:

three_starts = nice_rdd.takeSample(True, 3,1)
for random_words in three_starts:
    first,second = random_words[0]
    vals = random_words[1]
    phrase = [first,second]
    key = (phrase[-2],phrase[-1])
    num_words = 2
    while num_words < 21:
        thirds = [] # possible third words
        probs = [] # probabilities of third words
        num_thirds = float(len(vals))
        total = 0
        for word, count in vals:
            thirds.append(word)
            total += count
            probs.append(count/total)
        random_third = np.random.choice(thirds, 1,p=probs)
        phrase.append(random_third[0])
        key = (phrase[-2],phrase[-1])
        vals = nice_rdd.map(lambda x: x).lookup(key)[0]
        num_words += 1
    print ''.join([word + ' ' for word in phrase[:-1]]+[phrase[-1]])


# In[ ]:



