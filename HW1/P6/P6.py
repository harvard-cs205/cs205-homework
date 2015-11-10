from pyspark import SparkContext
import numpy as np
from itertools import chain
import random

sc = SparkContext("local[4]", appName="HW1-6 Markov Shakespeare")
sc.setLogLevel("ERROR")

txt = open("Shakespeare.txt").readlines()
txt = list(chain.from_iterable([line.split() for line in txt]))
txt2 = txt[1:] #Shift text by one word to get subsequent pairings
txt3 = txt[2:] #Shift text by two words to get third words to be appended to pairings

#Build the text pairings model

words1 = sc.parallelize(txt).filter(lambda x: not x.isdigit()).filter(lambda x: x != x.upper()).zipWithIndex().map(lambda x: (x[1],x[0])).partitionBy(20)
words2 = sc.parallelize(txt2).filter(lambda x: not x.isdigit()).filter(lambda x: x != x.upper()).zipWithIndex().map(lambda x: (x[1],x[0])).partitionBy(20)
words3 = sc.parallelize(txt3).filter(lambda x: not x.isdigit()).filter(lambda x: x != x.upper()).zipWithIndex().map(lambda x: (x[1],x[0])).partitionBy(20)

wordTuples = words1.join(words2).join(words3)
wordpairs_withfollowing = wordTuples.values() #Strip off index
wordpairs_agg = wordpairs_withfollowing.groupByKey().mapValues(list) #Group by word pairing
wordpairs_aggCounts = wordpairs_agg.mapValues(lambda l: [(w,l.count(w)) for w in set(list(l))]).cache() #Count occurence of each third word

	
#Given length of phrase (word_count), Shakespeare_generator will generate a phrase based on a biased mapping of word pairs
#Drawn from The Complete Works of William Shakespeare by The Project Gutenberg

def Shakespeare_generator(word_count):

    randPair = wordpairs_aggCounts.takeSample(False,1)[0][0] #Take random word pairing from Shakespeare text
    randSentence = [randPair[0]]
    randSentence.append(randPair[1])
    pair = (randPair[0],randPair[1]) #Create pair for lookup in Shakespeare text
    count = 2
    while count < word_count: #Iterate for length of word_count
        thirdOptions_seed = wordpairs_aggCounts.filter(lambda x: x[0] == pair).collect()[0][1] #Collect those words that follow directly after the selected pairing
        thirdOptions_list = list(chain.from_iterable([[w[0]] * w[1] for w in thirdOptions_seed])) #Create a biased list of third words
        third = random.choice(thirdOptions_list) #Choose random third word
        pair = (pair[1],third) 
        randSentence.append(third)
        count += 1
    
    return randSentence
	
#Generate random lines of text
	#Call Shakespeare_generator.py	
GeneratedText = []

for i_sentence in xrange(10):
	GeneratedText.append(Shakespeare_generator(20))
	
for i_sentence in xrange(len(GeneratedText)):
	print " ".join(GeneratedText[i_sentence]) + '\n'
