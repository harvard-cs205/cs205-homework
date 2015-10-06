import pyspark
from collections import defaultdict
from scipy import stats

### Helper functions ###

# Filter out any words that contain only numbers, contain only letters which are capitalized, or contains
# only letters which are capitalized and ends with a period.
def filterWords(word):
	return not (word.isupper() or word.isdigit()) and len(word) != 0

# Sample third word from empirical distribution function, biased according to count.
def sampleWord(thirdWords):
	sumCount = sum([word[1] for word in thirdWords])
	pk = [word[1]/float(sumCount) for word in thirdWords]
	xk = xrange(len(thirdWords))
	# Instantiate distribution and sample third word from distribution
	distr = stats.rv_discrete(name='distr', values=(xk,pk))
	return thirdWords[distr.rvs()][0]

# Write a phrase of numWords words given an rdd
def writePhrase(numWords, rdd):
	# Take a random element from rdd as beginning of phrase
	sample = rdd.takeSample(True, 1)
	phrase = []
	phrase.extend(sample[0][0])
	thirdWords = sample[0][1]

	# Repeatedly add third words by Markov Chain model
	while len(phrase) < numWords:
		thirdWords = rdd.map(lambda x: x).lookup((phrase[-2], phrase[-1]))
		# If last two words do not generate any third word, just stop here
		if thirdWords[0] == []:
			return " ".join(phrase)
		thirdWord = sampleWord(thirdWords[0])
		phrase.append(thirdWord)
	
	return " ".join(phrase)

# Reading Shakespeare.txt into Spark context
# sc = pyspark.SparkContext()
lines = sc.textFile('/Users/idzhang/COLLEGENOW/CS205/cs205-homework/HW1/P6/Shakespeare.txt')

# Splitting lines into words before filtering
words = lines.flatMap(lambda line: line.split(' ')).filter(filterWords) 

# Converts list of words to consecutive three word phrases
words = words.zipWithIndex().map(lambda (x,y): (y,x))
wordsShiftBy1 = words.map(lambda (y,x): (y-1, x))
wordsShiftBy2 = words.map(lambda (y,x): (y-2,x))

# Join the above 3 rdds, filtering out phrases with less than 3 words
threeWordPhrases = words.join(wordsShiftBy1).join(wordsShiftBy2)

# As desired, rdd contains ((word 1, word 2), [(word 3a, count 3a), (word 3b, count 3b), ...])
rdd = threeWordPhrases.map(lambda (idx, ((w1, w2), w3)): ((w1,w2,w3), 1)) \
	.reduceByKey(lambda count1,count2: count1+count2) \
	.map(lambda ((w1,w2,w3), count): ((w1,w2), [(w3,count)])) \
	.reduceByKey(lambda l1,l2: l1+l2)

# Search for phrases that start with "Now is ..."
rdd.map(lambda x: x).lookup((u'Now', u'is'))

# Print 10 generated phrases of 20 words each
for i in xrange(10):
	print writePhrase(20,rdd)
