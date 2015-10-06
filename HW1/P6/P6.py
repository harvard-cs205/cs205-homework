from pyspark import SparkContext
import random

# Filter words that are all capitalized or have all numbers 
# (also captures words with period at the end and capital letters)
def filter_word(word):
	return not (word.isupper() or word.isdigit())

# Grab word based on PDF of counts
def get_next_word(word_counts):
	counts = random.randint(1, sum([count for (word, count) in word_counts]))
	for word_count in word_counts:
		counts -= word_count[1]
		if counts <= 0:
			return word_count[0]

# Generate sentences from samples
def generate_sentences(rdd, samples):
	sentences = []
	for sample in samples:
		sentence = []

		# Grab words
		word1 = sample[0][0]
		word2 = sample[0][1]

		# Append words to our sentence array
		sentence.append(word1)
		sentence.append(word2)

		# Generate 20 more words for current sampling
		for x in xrange(18):
			# Lookup the last words of our sentence so far to get next word
			word_counts = rdd.map(lambda x: x) \
												.lookup((word1, word2))[0]

			# Get next word and append it to our sentence
			next_word = get_next_word(word_counts)
			sentence.append(next_word)

			# Reassign words
			word1 = word2
			word2 = next_word
		
		# Join words by space
		sentences.append(' '.join(sentence))

	return sentences

if __name__ == '__main__':
	sc = SparkContext()
	lines = sc.textFile('Shakespeare.txt')

	# Format words rdd
	words = lines.flatMap(lambda x: x.split()) \
								.filter(filter_word)

	# Index and shift words such that we can join the produced rdds to create an 
	# rdd of three-tuples of words in order
	index_words = words.zipWithIndex().map(lambda (word, id): (id, word))
	shift_words_1 = index_words.map(lambda (id, word): (id-1, word))
	shift_words_2 = index_words.map(lambda (id, word): (id-2, word))
	joined_words = index_words.join(shift_words_1).join(shift_words_2)
	word_mapping = joined_words.sortByKey() \
															.map(lambda (id, ((word1, word2), word3)): ((word1, word2, word3), 1)) \
															.reduceByKey(lambda x, y: x+y) \
															.map(lambda ((word1, word2, word3), count): ((word1, word2), [(word3, count)])) \
															.reduceByKey(lambda word_list1, word_list2: word_list1 + word_list2).cache()

  # Get 10 samples of words and print sentences produced
	samples = word_mapping.takeSample(False, 10)

	for sentence in generate_sentences(word_mapping, samples):
		print sentence
