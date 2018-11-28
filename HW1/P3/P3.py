import pyspark
from pyspark import SparkContext
sc = SparkContext("local[4]")
import numpy as np
sc.setLogLevel("ERROR")

#import words and then split up the letters in each word and alphabetize them
words = sc.textFile('words.txt')
wordcharacters = words.map(lambda word: np.sort(list(word)))

#now combine letters back together using join function. 
SortedLetterSequence = wordcharacters.map(lambda x: "".join(x))

#make two columns of our alphabetized words and .groupByKey to get anagrams
keyvaluerdd = words.map(lambda x: ("".join(np.sort(list(x))),x))
temp = keyvaluerdd.groupByKey().mapValues(list)
Anagrams = temp.map(lambda x: (x[0],(len(x[1]),x[1])))

#find maximum where we know max=11, so use filter.
maximum = Anagrams.filter(lambda x: x[1][0] == 11)

#there are three words with max=11. printing them out below.
print maximum.collect()[0]
print maximum.collect()[1]
print maximum.collect()[2]