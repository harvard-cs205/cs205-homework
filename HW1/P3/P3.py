#initializing/ importing all of the necessary libaries
#this was done with the help of Bryan Weinstein
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import seaborn as sns
sns.set_context('poster', font_scale=1.25)
from random import randint

# initializing spark
import pyspark as ps

config = ps.SparkConf()
config = config.setMaster('local[' + str(2*mp.cpu_count()) + ']')
config = config.setAppName('P3')

sc = ps.SparkContext(conf=config)

# importing the dictionary into an RDD
allWords=sc.textFile('EOWL_words.txt', use_unicode=True)

# taking the allWords RDD and sorting them into an array of keys and their respective words. 
# making sure there all of the sorted letters are unitque, by making sure all of the 
# anagrams are joined together if they are of the same key
wordMap = allWords.map(lambda (x): (''.join(sorted(x)), x)).groupByKey()

anagramRDD= wordMap.map(lambda x: (x[0], len(x[1]), x[1]))

result=anagramRDD.takeOrdered(1,lambda x: -x[1])

textFile=open('P3.txt','w')
for SortedLetterSequence, NumberOfValidAnagrams, Words in result:
	textFile.write("\n************ New Letter Sequence ****************\n")
	textFile.write("Sorted Letter Sequence: "+SortedLetterSequence+"\n")
	textFile.write("Number of Valid Anagrams: "+str(NumberOfValidAnagrams)+"\n")
	textFile.write("Words: ")
	for Word in Words:
		textFile.write(Word+", ")

textFile.close()

textFile=open('P3.txt','r')
print textFile.read()