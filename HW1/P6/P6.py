import numpy as np
import findspark
findspark.init()
import pyspark

sc = pyspark.SparkContext(appName='shakespeare')
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )


n_phrases = 10
phrase_length = 20

def main():
	words = load_txt()
	n_words = words.count()

	#  shift indices to create rdd of three-grams
	index_word = words.zipWithIndex().map(lambda (w, i): (i, w))
	offset_1 = index_word.map(lambda (i, w): (i-1, w))
	offset_2 = index_word.map(lambda (i, w): (i-2, w))
	joined = index_word.join(offset_1).join(offset_2)

	#  create rdd of the form (w1, w2, w3) : count
	reformat_join = lambda (i, ((w1, w2), w3)): ((w1, w2, w3), 1)
	three_grams = joined.map(reformat_join).reduceByKey(lambda x,y : x+y)

	#  create the rdd specified in the assignment description
	result = three_grams.map(lambda ((w1, w2, w3), count): ((w1, w2), (w3, count))).groupByKey()

	lookup = lambda k: list(result.map(lambda x: x).lookup(k)[0])

	for key in result.keys().takeSample(True, n_phrases):
		phrase = ' '.join(key)
		for j in range(phrase_length - 2):
			word_counts = lookup(key)
			phrase += ' ' + choose_word(word_counts)
			key = tuple(phrase.split()[-2:])
		print phrase


def choose_word(word_counts):
	"""Draw from a categorical distribution given the word_counts"""
	words, counts = zip(*word_counts)
	total = float(sum(counts))
	counts = [c / total for c in counts]
	return np.random.choice(words, size=1, p=counts)[0]


def valid_word(w):
	"True unless a word contains only numbers or only capital letters and periods"
	all_numbers = w.isdigit()
	all_upper = all([c.isupper() or c == '.' for c in w])
	return not (all_numbers or all_upper)

def load_txt():
	with open('Shakespeare.txt' , 'r') as rfile:
		words = sc.parallelize(rfile.read().split())
	words = words.filter(valid_word)
	return words

if __name__ == '__main__':
	main()