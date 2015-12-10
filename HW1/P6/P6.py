import re 
import random

word_filter = re.compile(r'^(\d+|[A-Z]+)(\.?)$')

def filter_words(word): 
	return word_filter.search(word) is None

text = sc.textFile('pg100.txt') 
words = text.flatMap(lambda line: line.split())
words_cleaned = words.filter(filter_words)

words_index1 = words_cleaned.zipWithIndex().map(lambda (K,V): (V,K))
words_index2 = words_index1.map(lambda (K,V): (K-1, V))
words_index3 = words_index2.map(lambda (K,V): (K-1, V))

triples = words_index1.join(words_index2).join(words_index3).map(
	lambda (K,V): (K, (V[0][0], V[0][1], V[1]))).sortByKey().cache()

triples_freq = triples.map(lambda (K,V): (V, 1)).reduceByKey(lambda x, y: x + y).cache()
results = triples_freq.map(lambda (K,V): ((K[0], K[1]), (K[2], V))).groupByKey().map(lambda (K,V): (K, tuple(V))).cache() 

next_word_lookup = dict(results.collect()) 

def nextWord(word_pair): 
	global next_word_lookup
	return random.choice(sum(([v]* wt for (v,wt) in next_word_lookup[word_pair]),[]))

starts = sc.parallelize(results.keys().takeSample(False, 10))

for _ in range(18): 
	starts = starts.map(lambda words: list(words).append(nextWord((words[-2], words[-1])))

results.take(10)
starts.take(10)