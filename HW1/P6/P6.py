from pyspark import SparkContext
import random

def removeUpper(word):
	count = 0
	for letter in word:
		if letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
			count = count + 1
	return not ((count==len(word)) or (count==len(word)-1 and word[-1]=='.'))


sc = SparkContext()
txt = sc.textFile('shakes.txt')
words = txt.flatMap(lambda w:w.split(' ')).filter(lambda w:not w.isdigit()).filter(removeUpper).collect()

threeWordsList = []
for i in range(len(words)-2):
	threeWordsList.append(tuple([words[i],words[i+1],words[i+2]]))
threeWordsList = sc.parallelize(threeWordsList)
threeWordsList = threeWordsList.map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y).map(lambda x:((x[0][0],x[0][1]),(x[0][2],x[1]))).groupByKey().map(lambda x:(x[0],list(x[1]))).cache()
totalWords = threeWordsList.count()

def getMaxWord(listOfWords):
	return max(listOfWords,key=lambda tup:tup[1])[0]

sentences = []
for i in range(10):
	x = threeWordsList.keys().collect()[random.randint(0,totalWords)]
	sentence = x[0] + " " + x[1]
	for i in range(18):
		newWord = getMaxWord(threeWordsList.lookup(x)[0])
		sentence += " " + newWord
		x = (x[1],newWord)
	sentences.append(sentence)
print sentences