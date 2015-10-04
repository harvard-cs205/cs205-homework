from pyspark import SparkContext
import random


def removeUpper(word):
	""" Removes uppercase and uppercase with ending period"""
	count = 0
	for letter in word:
		if letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
			count = count + 1
	return not ((count==len(word)) or (count==len(word)-1 and word[-1]=='.'))

def getMaxWord(listOfWords):
	"""Get the third word with the max number"""
	return max(listOfWords,key=lambda tup:tup[1])[0]

sc = SparkContext()
txt = sc.textFile('shakes.txt')
#Split sentences into words and remove numbers and uppercase words
words = txt.flatMap(lambda w:w.split(' ')).filter(lambda w:not w.isdigit()).filter(removeUpper).collect()
#Inefficient but works. Read in words with for loop, and put three connected words in a tuple. So ['a','b','c','d'] => [('a','b','c'),('b','c','d')]
threeWordsList = []
for i in range(len(words)-2):
	threeWordsList.append(tuple([words[i],words[i+1],words[i+2]]))
#parallelize the tuple list
threeWordsList = sc.parallelize(threeWordsList)
#('a','b','c') => (('a','b','c'),1)
threeWordsList = threeWordsList.map(lambda x:(x,1))
#Sum the tuples by key
threeWordsList = threeWordsList.reduceByKey(lambda x,y:x+y)
#(('a','b','c'),8) = > (('a','b'),('c',8))
threeWordsList = threeWordsList.map(lambda x:((x[0][0],x[0][1]),(x[0][2],x[1])))
# [(('a','b'),('c',8)),(('a','b'),('d',4))] => [(('a','b'),[('c',8),('d',4)])]
threeWordsList = threeWordsList.groupByKey().map(lambda x:(x[0],list(x[1]))).cache()
totalWords = threeWordsList.count()

sentences = []
for i in range(10):
	#Pick first random tuple i.e. ('She','looked')
	x = threeWordsList.keys().collect()[random.randint(0,totalWords)]
	#Initialize sentence
	sentence = x[0] + " " + x[1]
	#Get the next 18 words of sentence
	for i in range(18):
		#Look up ('She','looked') and get the max word in its list 
		newWord = getMaxWord(threeWordsList.lookup(x)[0])
		sentence += " " + newWord
		x = (x[1],newWord)
	sentences.append(sentence)
print sentences


