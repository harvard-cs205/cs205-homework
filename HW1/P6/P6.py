from __future__ import division
import numpy as np
import pyspark

sc = pyspark.SparkContext(appName = "Spark1")

allWords = sc.textFile('shakespeare.txt') # I copied the text file to my local directory and deleted the beginning and ending legal non-Shakespeare business

allWords = allWords.map(lambda x: np.array(x.split(' '))).flatMap(lambda x: x[x != '']) # split text by line then remove unnecessary white spaces

allWords = allWords.filter(lambda x: x.isupper() == False) # Filter for all caps and all numbers
allWords = allWords.filter(lambda x: x.isdigit() == False)

allWords = allWords.zipWithIndex().map(lambda (n,id): (id,n)).partitionBy(10).cache() # index words

nextWords = allWords.filter(lambda x: x[0] > 0).map(lambda x: (x[0]-1,x[1])).partitionBy(10) 

wordTriples = allWords.join(nextWords).map(lambda x: (x[0]+2,(x[1][0],x[1][1]))).partitionBy(10).cache() # form triples here and below
wordTriples = allWords.join(wordTriples).map(lambda x: (x[1][1],x[1][0])) # ((first, second),third)

wordTriples = wordTriples.groupByKey().mapValues(list)
wordTriples = wordTriples.map(lambda x: (x[0],[(ii,sum(np.array(x[1])==ii)) for ii in np.unique(x[1])])).partitionBy(10).cache()

def getNextWord(orderedPair): # calculate random third word based on empirical counts
	counts = np.array([ii[1] for ii in orderedPair[1]])
	probs = np.cumsum(counts)/sum(counts)
	randI = np.random.uniform(0,1,1)
	ind = sum(probs < randI)
	return (orderedPair[0],[orderedPair[1][ind][0]])

initial = wordTriples.takeSample(False,10,1) # initialize 10 sentences

sentences = sc.parallelize(initial)
sentences = sentences.map(lambda x: (x[0],[x[0][0], x[0][1]])).partitionBy(10).cache() # get first 2 words, use last 2 words as key

for ii in range(0,18): # generate 18 more words for each of the 10 sentences
	last2 = sentences.join(wordTriples).map(lambda x: (x[0],x[1][1]))
	last2 = last2.map(getNextWord).partitionBy(10)
	sentences = sentences.join(last2).map(lambda x: (x[0],x[1][0]+x[1][1]))
	sentences = sentences.map(lambda x: ((x[1][-2],x[1][-1]),x[1])).partitionBy(10).cache() # update key to be last 2 words

sentences = sentences.map(lambda x: x[1])
print sentences.collect()

# code below was to verify hashing of random data
"""
aa = sc.parallelize([('ae',1),('bf',1),('cz',1),('do',1),('ep',1),('f2',1),('g6',1),('h7',1),('i7',1),('j6',1)]).partitionBy(10)
bb = sc.parallelize([('cz',2),('i7',2),('ep',2),('f2',2),('bf',2),('h7',2),('ae',2),('j6',2),('g6',2),('do',2)]).partitionBy(10)

aa2 = sc.parallelize([('ae',1),('bf',1),('cz',1),('do',1),('ep',1),('f2',1),('g6',1),('h7',1),('i7',1),('j6',1)])
bb2 = sc.parallelize([('cz',2),('i7',2),('ep',2),('f2',2),('bf',2),('h7',2),('ae',2),('j6',2),('g6',2),('do',2)])

print aa.join(bb).collect()
print aa2.join(bb).collect()
print aa.glom().collect()
print bb.glom().collect()
"""
















