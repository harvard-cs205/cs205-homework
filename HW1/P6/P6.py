import random
sc.setLogLevel('ERROR')

shake = sc.textFile('pg100.txt')

words = shake.flatMap(lambda line : line.split())

def isnotInt(word):
	try:
		int(word)
	except:
		return True
	else:
		return False

# filter out all patters as requested
words = words.filter(isnotInt)
words = words.filter(lambda word: not word.isupper())
words = words.filter(lambda word: not (word[:-1].isupper() and word[-1] == '.'))

# make indicies with consecutive words
words1 = words.zipWithIndex().map(lambda (K,V) : (V,K))
words2 = words1.map(lambda (K,V) : (K+1,V))
words3 = words1.map(lambda (K,V) : (K+2,V))

# get three words in a row
triple = words3.join(words2).join(words1).sortByKey()
triple = triple.map(lambda (K,V) : ((V[0][0], V[0][1], V[1]), 1))

# get frequency of the phrase
freq = triple.reduceByKey(lambda x,y : x + y)

# make only first 2 words the key, have following words & freq's as value
pairs = freq.map(lambda (K,V) : ((K[0],K[1]), (K[2],V))).groupByKey()
pairs = pairs.map(lambda (K,V): (K,list(V))).cache()

# generate phrase 

allwords = sc.broadcast(dict(pairs.collect()))
# helper function to get third word, wieghted by frequency
def nextWord (wordpair):
	thirdWords = allwords.value[wordpair]
	wordlist = []
	for (word, count) in thirdWords:
		for i in range(count):
			wordlist.append(word)
	return random.choice(wordlist)

# get random sample of 10 pairs, make Key the next pair, Value the return phrase
startPhrases = pairs.takeSample(False, 10)
phrases = sc.parallelize(startPhrases).map(lambda (K,V) : [K[0],K[1]])

for _ in range(18):
	phrases = phrases.map(lambda words : words + [nextWord((words[-2],words[-1]))])

stringphrases = phrases.map(lambda wds : ' '.join(wds)).take(10)
print stringphrases

