import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext()

from collections import Counter # for 3rd word counts
import numpy as np 				# for sampling
import time 					# for benchmarking

def quiet_logs(sc):
	''' Shuts down log printouts during execution (thanks Ray!) '''
	logger = sc._jvm.org.apache.log4j
	logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)
	logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)
	logger.LogManager.getLogger("amazonaws").setLevel(logger.Level.WARN)

quiet_logs(sc)


''''''''''''''''''''''''''
''' format word corpus '''
''''''''''''''''''''''''''
n_part      = 10 # num partitions
shakespeare = sc.textFile("Shakespeare.txt", n_part)

def excluded(w):
	''' checks if word meets exclusion criteria ''' # exclude if:
	all_upper = w.isupper() 						# all uppercase
	all_upper_plus_period = False					
	if w[-1]==".": 
		all_upper_plus_period = w[:-1].isupper()	# all uppercase ending in period
	all_numbers = w.isdigit()						# all digits
	return all_upper or all_upper_plus_period or all_numbers

def format_corpus():
	''' splits corpus into words, filters by exclusion criteria'''
	return lambda corpus: [word for word in corpus.split(" ") if word and not excluded(word)]

def countit(wlist):
	''' returns (word,frequency) tuple '''
	return Counter(wlist).most_common()

''' we want our corpus in the format: ( (word1,word2), [(word3a,word3a_freq),(word3b,word3b_freq),...]) 
	- we can do this by first getting ( (word1,word2), word3 ) tuples.
	- make 3 corpora: first corpus unchanged, second corpus offset start index by 1,  third corpus offset start by 2
	- then join all 3 corpora into one - this gives us ( (word1,word2), word3 ) tuples. 
	- from there we can groupByKey and format.
'''

words  = shakespeare.flatMap(format_corpus(), preservesPartitioning=True).cache()
wct    = words.count()
wlist1 = words.zipWithIndex().filter( lambda x: x[1] < wct-2 ).map( lambda x: tuple(reversed(x)) ).partitionBy(n_part).cache()
wlist2 = words.zipWithIndex().filter( lambda x: (x[1] > 0) and (x[1] < wct-1) ).map( lambda x: tuple(reversed(x)) ).map( lambda x: (x[0]-1,x[1]) ).partitionBy(n_part).cache()
wlist3 = words.zipWithIndex().filter( lambda x: x[1] > 1 ).map( lambda x: tuple(reversed(x)) ).map( lambda x: (x[0]-2,x[1]) ).partitionBy(n_part).cache()
rdd    = (wlist1.join(wlist2)
				.join(wlist3)
				.map(lambda x: x[1])
				.groupByKey()
				.map(lambda x: ( x[0],countit(x[1]) ))
				.cache()
	   )
	
''''''''''''''''''''
''' markov model '''
''''''''''''''''''''

''' Note: Here are 2 versions of the Markov model: RDD-only and RDD/normal Python hybrid '''

''' COMMON SETUP FOR BOTH IMPLEMENTATIONS '''

def get_third(third_words):
	''' creates empirical PMF from 3rd word counts, draws one word by sampling '''
	words,freqs = zip(*third_words)
	pmf         = np.array(freqs)/float(sum(freqs)) # PMF for observations
	third       = np.random.choice(words,p=pmf)     # draw word from PMF
	return third

n_samp = 10
n_iter = 20
starts = rdd.takeSample(False,n_samp)       # draw n_samp (first_word,second_word) tuples


''' PURE RDD IMPLEMENTATION '''
def start_filter(s):
	return lambda x: x[0] in [start[0] for start in s]

def run_pure():
	timeA=time.time()

	start  = rdd.filter( start_filter(starts) ) # filter rdd down to 10 starts
	phrase = (start.zipWithIndex()              # initialize phrase as (idx, (word1,word2))
				   .map( lambda x: tuple(reversed(x)) )
				   .map( lambda x: (x[0],[el for el in x[1][0]]) )
				   .partitionBy(n_part)
			  )

	for i in range(n_iter):   # we want n_iter-word phrases (target:20)
		if i:                 # after initial round, build start based on prior keys
			start = (rdd.join(key.map(lambda x: (x,['',0]))) # add dummy [(word3,freq)] to key for join
						 .partitionBy(n_part)
						 .map(lambda x: (x[0],x[1][0]))      # drop dummy (necessary?)
						 .partitionBy(n_part)
						 .cache()
					)
			
		key    = (start.map(lambda x: (x[0][1], get_third(x[1]))) # draw word3 from PMF, build new key
					   .partitionBy(n_part)
				  )
		phrase = (phrase.union( key.zipWithIndex()                      # union new word3 to phrase 
								   .map( lambda x: tuple(reversed(x)) ) # make index into key for union
								   .map( lambda x: (x[0],x[1][1]) )     # take only word3
								   .partitionBy(n_part) 
							  )
						.reduceByKey(lambda a,b: a+[b])                 # reduce to list of phrase words
						.partitionBy(n_part)
				)
		
	print phrase.map(lambda x: ' '.join(x[1])).collect() # print each phrase as string (space separator)

	timeB=time.time()
	print "total time (pure):",timeB-timeA

run_pure() # ~293s


''' HYBRID RDD/PYTHON IMPLEMENTATION '''
def run_hybrid():
	timeA=time.time()

	for start in starts:
		key    = start[0]
		phrase = [key[0],key[1]]
		for i in range(n_iter): # we want n_iter-word phrases (target:20)
			second = key[1] 
			third  = get_third(rdd.lookup(key)[0]) 	# use new (first,second) to get 3rd word draw
			key    = (second,third)					# make new (first,second) from current (second,third)
			phrase.append(third)  					# append drawn 3rd to phrase
		print ' '.join(phrase)

	timeB=time.time()
	print "total time (hybrid):",timeB-timeA

run_hybrid() # ~210s