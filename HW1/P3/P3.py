from __future__ import division
import numpy as np
import pyspark

sc = pyspark.SparkContext(appName = "Spark1")

"""
aa = sc.parallelize([(1,'hi'),(1,'oh')])
aa = aa.groupByKey().mapValues(list)
print aa.collect()
bb = sc.parallelize([(1,'an'),(1,'be')])
bb = bb.groupByKey().mapValues(list)
cc = aa.join(bb)
print cc.collect()
cc = cc.map(lambda x: (x[0],[item for sublist in x[1] for item in sublist]))
cc = sc.parallelize(cc.map(lambda x: x[1]).collect()[0])
print cc.collect()
print cc.take(2)
"""

allWords = sc.textFile('A Words.csv').cache()
allWords = allWords.map(lambda x: (1,x))
allWords = allWords.groupByKey().mapValues(list)
bWords = sc.textFile('B Words.csv').cache()
bWords = bWords.map(lambda x: (1,x))
bWords = bWords.groupByKey().mapValues(list)
allWords = allWords.join(bWords)
allWords = allWords.map(lambda x: (x[0],[item for sublist in x[1] for item in sublist])) # read in data below
fileNames = ['C Words.csv','D Words.csv','E Words.csv','F Words.csv','G Words.csv','H Words.csv','I Words.csv','J Words.csv','K Words.csv',\
'L Words.csv','M Words.csv','N Words.csv','O Words.csv','P Words.csv','Q Words.csv','R Words.csv','S Words.csv','T Words.csv','U Words.csv',\
'V Words.csv','W Words.csv','X Words.csv','Y Words.csv','Z Words.csv',]
for ii in range(0,24):
	nextWords = sc.textFile(fileNames[ii]).cache()
	nextWords = nextWords.map(lambda x: (1,x))
	nextWords = nextWords.groupByKey().mapValues(list)
	allWords = allWords.join(nextWords)
	allWords = allWords.map(lambda x: (x[0],[item for sublist in x[1] for item in sublist]))

allWords = sc.parallelize(allWords.map(lambda x: x[1]).collect()[0])
print len(allWords.collect())
allWords = allWords.map(lambda x: (''.join(sorted(x)),x)) # make an RDD where key is word in sorted order
allWords = allWords.groupByKey().mapValues(list) # group by key to get anagrams
allWords = allWords.map(lambda x: (x[0],len(x[1]),x[1])) # get desired output
gramCount = allWords.map(lambda x: x[1])
maxCount = gramCount.max()
maxEntry = allWords.filter(lambda x: x[1] == maxCount) # get entry with most anagrams
print maxEntry.collect()


