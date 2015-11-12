from pyspark import SparkContext
import numpy as np

# Function to load in text and filter as per the spec
def loadAndFilter(sc, filename):
	with open(filename, 'r') as txtfile:
		unfiltWords = sc.parallelize(txtfile.read().split())
	# Filter out words of all nums
	wordsNoNums = unfiltWords.filter(lambda word: not word.isdigit())
	# Filter out words of all caps
	wordsNoCaps = wordsNoNums.filter(lambda word: not word.isupper())
	# Filter out words of all caps ending with a period 
	words = wordsNoCaps.filter(lambda word: not word.isupper() and not word[:-1] == '.')
	return words

# Function to build the 3grams and to return them in the format of the spec
def build3Grams(sc, filename):
	filteredWords = loadAndFilter(sc, filename)

	# Neat little trick to get 3-grams to build. I collaborated on this part with 
	# Andrew Mauboussin and used it as I thought it was very clever.
	# This code creates an index for each word and creates 3 list, each with an index
	# that is offset by 1 from the previous. When these are joined, we have 3-grams!
	indexedWords = filteredWords.zipWithIndex().map(lambda (word, i): (i, word))
	offsetIndexedWords = indexedWords.map(lambda (i, word): (i-1, word))
	offsetIndexedWords2 = offsetIndexedWords.map(lambda (i, word): (i-1, word))

	# Make sure to take only the values to get rid of the index
	threeGrams = indexedWords.join(offsetIndexedWords).join(offsetIndexedWords2).values()

	# Create a count for each 3gram and reduce to compute total counts for each
	# ((word1, word2), word3), count)
	threeGramsCounts = threeGrams.map(lambda x: (x, 1)).reduceByKey(lambda c1, c2: c1 + c2)
	
	# Format 3grams according to spec
	# ((word1, word2), (word3, count))
	threeGramsCountsFormat = threeGramsCounts.map(lambda (((w1, w2), w3), count): ((w1, w2), (w3, count)))

	# Group by each pair and apply a list map to get around annoying ResultIterable
	threeGramsCountsMerged = threeGramsCountsFormat.groupByKey().map(lambda x: (x[0], list(x[1])))
	return threeGramsCountsMerged

# Function to actually build the sentences given number of 
# sentences, length of each, and an RDD of words
def buildSentences(n, l, rdd):
	sentences = []
	for start in rdd.takeSample(True, n):
		# Kind of nasty list comprehensions here
		# start = ((word1, word2), [(word3, count), ...])
		# sentence = "word1 word2"
		sentence = "" + start[0][0] + " " + start[0][1]
		for word in range(l-2):
			# Lookup (word1, word2). Redundant map to get around Spark 1.5.0 lookup bug
			twoGram = list(rdd.map(lambda x: x).lookup(start[0]))

			# Here, words = [word3, word3, ...]. counts = [count, count, ...]
			words = [w[0] for w in twoGram[0]]
			counts = [w[1] for w in twoGram[0]]

			# Compute probabilities and select next word to append to sentence
			totalProba = float(sum(counts))
			allProbas = [proba/totalProba for proba in counts]
			nextWord = np.random.choice(words, p=allProbas)
			sentence = sentence + " " + nextWord

			# Bit hacky, but the code expects start to have a list of follow-up
			# words, so an empty list will suffice as a placeholder
			# start = ((word2, word3), [])
			start = ((start[0][1], nextWord), [])
		sentences.append(sentence)
	return sentences

# Given params for assignment
numSentences = 10
lenSentences = 20

# Set up context and filename
sc = SparkContext("local", "P6")
filename = "shakespeare.txt"

# Set up RDD and test according to the spec
rdd = build3Grams(sc, filename)
#print rdd.map(lambda x: x).lookup(('Now', 'is'))

# Pretty print the sentences
sentences = buildSentences(numSentences, lenSentences, rdd)
# print sentences
for s in sentences:
	print s


