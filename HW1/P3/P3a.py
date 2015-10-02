import pyspark

sc = pyspark.SparkContext()
wlist = sc.textFile('/Users/idzhang/COLLEGENOW/CS205/EOWL-v1.1.2/CSV/')

# Function that takes in a word and returns sortedLetterSequence
def sortLetterSequence(word):
	return ''.join(sorted(word))

# Pairs each word with its sorted letter sequence, and swaps them
wordWithSort = wlist.map(lambda word: (word, sortLetterSequence(word))).map(lambda (x,y): (y,x))

# Gather all words (values) corresponding to same sorted letter sequence (key) in a list
sortWithWords = wordWithSort.mapValues(lambda x: [x]).reduceByKey(lambda a,b: a+b)

# Add number of valid anagrams
rddWithAddedCount = sortWithWords.map(lambda (sorted, wordList): (sorted, len(wordList), wordList))

# Extract and print line from RDD with largest number of valid anagrams
lineWithLargestCount = rddWithAddedCount.takeOrdered(1, lambda x: -x[1])
print lineWithLargestCount