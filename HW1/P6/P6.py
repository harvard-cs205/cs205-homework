import pyspark
from pyspark import SparkContext
import time
import random
from random import randint

f = open('P6.txt', 'w')

def isNotNumber(word):
	try:
		int(word)
		return False
	except ValueError:
		return True

#take a list and then return a list
#of (word, num times word appears in list)

def listToCounts(lst):
	results = []
	dic = {}
	for word in lst:
		if word in dic:
			dic[word] += 1
		else:
			dic[word] = 1
	for word in lst:
		results.append((word, dic[word]))
	return list(set(results))

#Determine which list is the word-pair
#and set the other one to be the listed "third word"
#Used in a map function later. 
def incorporateThirdWord(x, y):
	if len(x) == 2:
		return (x, [y])
	else: 
		return (y, [x])

sc = SparkContext()
sc.setLogLevel('ERROR')
#sc.parallelize('s3://Harvard-CS205/Shakespeare/Shakespeare.txt')
rawText = sc.textFile('Shakespeare.txt')

#Clean the data to make sure it doesn't have the concerns listed
#in the problem statement. 
words = rawText.flatMap(lambda x: x.split(' '))
words = words.filter(lambda x: len(x) > 0)
noNumbers = words.filter(lambda x: False if x.isdigit() else True)
noCaps = noNumbers.filter(lambda x: False if x.isupper() else True)
noCapsPeriod = noCaps.filter(lambda x: False if (x[:len(x)-1].isupper() and x[-1] is '.') else True)

#index these words so we can go ahead and get a feeling of which ones come first
#or second.
text = noCapsPeriod.zipWithIndex()

#The text, normally with the word first, and then the index of the word as the value
text = text.map(lambda x: (x[1], (x[0])))
#Shift all these back one. So now, index "n" in text is the "nth" word in Shakespeare
#and #n# in one_shift is the "n+1"th word in Shakespeare
one_shift = text.map(lambda (x, y): (x-1, y))
two_shift = text.map(lambda (x, y): (x-2, y)).sortByKey()

#Join and reduce by key so we can get pairs of words. For some reason
#Reduce by keys messes with the order, so sort it afterwards. 
all_shifts = text.join(one_shift)
all_shifts = all_shifts.reduceByKey(lambda x, y: x + y).sortByKey()
all_shifts.cache()

#Add in the third word (aka two shift, which is the third word)
shifts_and_next_word = all_shifts.join(two_shift).sortByKey()

#Use the function defined above to incorporate the third word. 
shifts_and_next_word = shifts_and_next_word.reduceByKey(lambda x, y: incorporateThirdWord(x, y)).sortByKey()

#Combine the values and make the latter into a list.
shifts_and_next_word = shifts_and_next_word.mapValues(lambda (x, y): (x, [y]))

#get the values since that is all we care about
word_values = shifts_and_next_word.values()

final = word_values.reduceByKey(lambda x, y: x + y).sortByKey()

#Go from list to counts and now we have our finalized rdd, "final"
final = final.mapValues(lambda y: listToCounts(y))

#final.take(500)

# print final.lookup(('and', 'with'))

def numWords(s):
	return len(s.split(' '))

# Generate sentence works by first taking a random sample from the RDD
# it's passed in as an argument. It then gets the first and second word
# of that RDD. Then it starts constructing a sentence from that seed.
# Now, while the length of that sentence is not 20 yet, we create
# an empirical PMF of the possible words that go next and then
# randomly sample from that PMF. We then use the preceding two words
# as our new pair to create our next PMF and the continue the process. 

def generateSentence(rdd):
	start = rdd.takeSample(True, 1)[0]
	sentence = ''
	firstWord = start[0][0]
	secondWord = start[0][1]
	sentence += firstWord
	sentence += ' '
	sentence += secondWord
	while numWords(sentence) <= 20:
		PMF = []
		theRest = rdd.lookup((firstWord, secondWord))[0]
		for char, num in theRest:
			for i in xrange(num):
				PMF.append(char)
		r = randint(0, len(PMF)-1)
		newWord = PMF[r]
		sentence += newWord
		sentence += ' '
		firstWord = secondWord
		secondWord = newWord
	return sentence

for i in xrange(10):
	print >> f, generateSentence(final)





