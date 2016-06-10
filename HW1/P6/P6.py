import findspark
findspark.init()
import pyspark
import time
import re
from random import random
from bisect import bisect
start = time.time()
sc = pyspark.SparkContext(appName='test')

def mapFunc(word):
	wordList = []
	word = word.encode('utf-8')
	wordList = re.split(" ", word)
	return wordList


def filterFunc(word):
	if len(word) == 0:
		return False
	return (not word[:-1].isupper()) if word[-1] == '.' else  not word.isupper() and not word.isdigit()

textRDD = sc.textFile('pg100.txt',8)
rawRDD = textRDD.flatMap(mapFunc)
wordsRDD = rawRDD.filter(filterFunc) # Now wordsRDD contains all the desired words

firstWords, secondWords, thirdWords = wordsRDD.zipWithIndex().map(lambda (k, v): (v, k)), wordsRDD.zipWithIndex().map(lambda (k, v):(v-1,k)).filter(lambda (k, v): k >= 0), wordsRDD.zipWithIndex().map(lambda (k, v): (v-2, k)).filter(lambda (k, v):k >= 0)
threeWordSet = firstWords.join(secondWords).join(thirdWords).map(lambda (k, v): v)

zipSameWord = threeWordSet.groupByKey().map(lambda (k, v): (k, (list(v))))
zipSameWordDict = zipSameWord.map(lambda (k, v): (k,dict((x,v.count(x)) for x in v)))
keyWord = ('Now', 'is')
resList = zipSameWordDict.map(lambda x: x).lookup(keyWord)
res = resList[0].items()

def weighted_choice(choices):
    values, weights = zip(*choices)
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = random() * total
    i = bisect(cum_weights, x)
    return values[i]


randWords = zipSameWordDict.takeSample(withReplacement = 1, num = 1)[0][0]
randSeedWords = (randWords[0],randWords[1])
for i in range(10):
	randWords = zipSameWordDict.takeSample(withReplacement = 1, num = 1)[0][0]
	sentence = []
	randSeedWords = (randWords[0],randWords[1])
	sentence.append(randSeedWords[0])
	sentence.append(randSeedWords[1])
	seedWords = zipSameWordDict.map(lambda x: x).lookup(randSeedWords)
	thirdWord = weighted_choice(seedWords[0].items())
	keyWord = (randSeedWords[1], thirdWord)
	sentence.append(thirdWord)
	for j in range(3, 20):
		tmpWordChoice = zipSameWordDict.map(lambda x: x).lookup(keyWord)
		tmpThirdWord = weighted_choice(tmpWordChoice[0].items())
		sentence.append(tmpThirdWord)
		keyWord = (keyWord[1],tmpThirdWord)
	with open('P6.txt', 'a') as log_file:
		log_file.write(' '.join(sentence)+'\n\n')
	print ' '.join(sentence),'\n'
log_file.close()




