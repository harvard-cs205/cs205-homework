import findspark
findspark.init()

import pyspark
import numpy as np

sc = pyspark.SparkContext()

# make pyspark shut up
sc.setLogLevel('WARN')


def getCounts(lst):
	dct = {}
	result = []
	for word in lst:
		if word not in dct:
			dct[word] = 1
		else:
			dct[word] += 1
	for word in dct:
		result.append((word, dct[word]))
	return result

def getNextWord(word1, word2, rdd):
	lstOfCounts = rdd.lookup((word1, word2))[0]
	words = [word for (word, count) in lstOfCounts]
	counts = [count for (word, count) in lstOfCounts]
	total = sum(counts)
	probabilities = [float(count)/total for count in counts]
	return np.random.choice(words, p=probabilities)


N =16

lines = sc.textFile("pg100.txt", N)
indexed_lines = lines.zipWithIndex()

start = 173
end = 124368

shakespeare_lines = indexed_lines.filter(lambda (u,v): v > start and v < end)
words = shakespeare_lines.keys().flatMap(lambda line: line.split())
filtered_words = words.filter(lambda word: not(word.isdigit())\
 					and not(word.isalpha() and word.isupper())\
 					and not(word[:-1].isalpha() and word[:-1].isupper and word[-1] == '.'))

indexed_words = filtered_words.zipWithIndex().map(lambda (word, index): (index, word))
offset1 = indexed_words.map(lambda (index, word) : (index-1, word))
offset2 = offset1.map(lambda (index, word) : (index-1, word))

triplet = indexed_words.join(offset1, N).join(offset2, N)

# get format ((Word1, Word2), [(Word3a, Count3a), (Word3b, Count3b), ...]), map x to x because of spark bug
rdd = triplet.values().groupByKey().mapValues(list).mapValues(getCounts).map(lambda x:x).cache()

sentences = []

starts = rdd.takeSample(False, 10)

for start in starts:
	sentence = []
	sentence.append(start[0][0])
	sentence.append(start[0][1])
	for i in range(2, 20):
		sentence.append(getNextWord(sentence[-2], sentence[-1], rdd))
	sentences.append(' '.join(sentence))

for sentence in sentences:
	print sentence
	print










