from pyspark import SparkContext
import numpy as np

sc = SparkContext("local", "HW1-3 Anagrams")

#Reading in words from locally saved version of the online list
wList = sc.textFile('words.txt')

#Creates tuple of alphabetized list from the letters in each word
letteredWords = wList.map(lambda w: (sorted(list(w)),w)) 

#Creates key from alphabetized letters and a list of valid words using that letter
orgJumble = letteredWords.map(lambda w: (''.join(w[0]),[w[1]]))

#Combines all lists made from each alphabetized key
combJumble = orgJumble.reduceByKey(lambda x, y: x+y)

#Sort based on length of valid word list
sortedCombJumble = combJumble.sortBy(lambda KV: len(KV[1]), ascend=False)

print sortedCombJumble.take(1)