__author__ = 'xiaowen'
import findspark
findspark.init('/home/xiaowen/spark')
import pyspark
from collections import Counter
import numpy as np


# build sparkcontext object
sc = pyspark.SparkContext(appName="P6")

# read data, convert it to RDD
lines = sc.textFile('/home/xiaowen/cs205-homework/HW1/P6/pg100.txt')

# split words
wlist_before = lines.flatMap(lambda x:  x.split())

# filter out words only contain numbers, only capitalized letters or only capitalized end with a period
wlist = wlist_before.filter(lambda x: not(x.isdigit() or x.isupper() or (x[-1] == '.' and x[:-1].isupper())))

# get the length of wlist
length = len(wlist.collect())

# zip wlist with index and make index as key
withindex = wlist.zipWithIndex()

# Create three words list RDD
wlist_1 = withindex.map(lambda x: (x[1], x[0])).filter(lambda x: x[0] < length-2).partitionBy(4)
wlist_2 = withindex.map(lambda x: (x[1]-1, x[0])).filter(lambda x: x[0] < length-1).partitionBy(4)
wlist_3 = withindex.map(lambda x: (x[1]-2, x[0])).filter(lambda x: x[0] < length).partitionBy(4)

# join the three RDD together and create a pairs RDD where k is (word1,word2), v is word3
w_3d = wlist_1.join(wlist_2).join(wlist_3)
# check if w_3d nd wlist_1 are co-partitioned
assert (w_3d.partitioner == wlist_1.partitioner)
w_3d = w_3d.map(lambda x: x[1])

# group w_3d by key and transform it to ((word1,word2),[(word3a,count3a),(word3b,count3b),...])
rdd_3d = w_3d.groupByKey().map(lambda x: (x[0], list(x[1]))).sortByKey()
rdd_3d_cache = rdd_3d.cache()
rdd_3d_count = rdd_3d.map(lambda x: (x[0], Counter(x[1])))


# function to generate phrases of 20 words
def generate_words(x, rdd, n):
    words = list(x)
    for i in range(n-2):
        word = np.random.choice(rdd.lookup((words[i], words[i+1]))[0])
        words.append(word)
    return ' '.join(words[:])

# randomly choose 10 random (word1,word2)
random_10 = rdd_3d.takeSample(True, 10, 1)

# use generated (word1,word2) to start with and generate 10 random phrases of 20 words
for i in random_10:
    print generate_words(i[0], rdd_3d_cache, 20)





