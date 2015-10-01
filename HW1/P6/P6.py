import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext()

from collections import Counter
import numpy as np

def quiet_logs(sc):
	''' Shuts down log printouts during execution (thanks Ray!) '''
	logger = sc._jvm.org.apache.log4j
	logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)
	logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)
	logger.LogManager.getLogger("amazonaws").setLevel(logger.Level.WARN)

quiet_logs(sc)

''' format word corpus '''
part_size   = 4 # num partitions
shakespeare = sc.textFile("Shakespeare.txt", part_size)

def excluded(w):
	''' checks if word meets exclusion criteria '''
	all_upper = w.isupper() 						# all uppercase?
	all_upper_plus_period = False					
	if w[-1]==".": 
		all_upper_plus_period = w[:-1].isupper()	# all uppercase ending in period?
	all_numbers = w.isdigit()						# all digits?
	return all_upper or all_upper_plus_period or all_numbers

def format_corpus():
	''' splits corpus into words, filters by exclusion criteria'''
	return lambda corpus: [word for word in corpus.split(" ") if word and not excluded(word)]

words = shakespeare.flatMap(format_corpus(), preservesPartitioning=True).cache()

def countit(wlist):
	''' returns (word,frequency) tuple '''
	return Counter(wlist).most_common()

''' we want our corpus in the format: ( (word1,word2), [(word3a,word3a_freq),(word3b,word3b_freq),...]) 
	- we can do this by first getting ( (word1,word2), word3 ) tuples.
	- make 3 corpora: first corpus unchanged, second corpus offset start index by 1,  third corpus offset start by 2
	- then join all 3 corpora into one - this gives us ( (word1,word2), word3 ) tuples. 
	- from there we can groupByKey and format.
'''

wct    = words.count()
wlist1 = words.zipWithIndex().filter( lambda x: x[1] < wct-2 ).map( lambda x: tuple(reversed(x)) ).partitionBy(part_size).cache()
wlist2 = words.zipWithIndex().filter( lambda x: (x[1] > 0) and (x[1] < wct-1) ).map( lambda x: tuple(reversed(x)) ).map( lambda x: (x[0]-1,x[1]) ).partitionBy(part_size).cache()
wlist3 = words.zipWithIndex().filter( lambda x: x[1] > 1 ).map( lambda x: tuple(reversed(x)) ).map( lambda x: (x[0]-2,x[1]) ).partitionBy(part_size).cache()
rdd    = (wlist1.join(wlist2)
				.join(wlist3)
				.map(lambda x: x[1])
				.groupByKey()
				.map(lambda x: ( x[0],countit(x[1]) ))
				.cache()
	   )
	

''' markov model '''

def get_third(third_words):
	''' creates empirical PMF from 3rd word counts, draws one word by sampling '''
	words,freqs = zip(*third_words)
	pmf         = np.array(freqs)/float(sum(freqs)) # PMF for observations
	third       = np.random.choice(words,p=pmf)     # draw word from PMF
	return third

starts = rdd.takeSample(False,10) # draw 10 random (first_word,second_word) tuples

for start in starts:
	key    = start[0]
	phrase = [key[0],key[1]]
	for i in range(20): # we want 20-word phrases
		second = key[1] 
		third  = get_third(rdd.lookup(key)[0]) # use new (first,second) to get 3rd word draw
		key    = (second,third)		# make new (first,second) from current (second,third)
		phrase.append(third)  # append drawn 3rd to phrase
	print ' '.join(phrase)


