from pyspark import SparkContext
sc = SparkContext(appName="P3")
import numpy.random as rand

# Open file and split into words

with open('shakespeare.txt') as read_file:
	read = read_file.read()
	text = read.replace('\n',' ').split()
	
# Get rid of numbers, all caps words, and all caps words ending with a period

def isnum(w):
	"""
	Checks if w is a number.
	"""
	try:
		float(w)
		return True
	except:
		return False
	
def isallupper(x):
	"""
	Just like str.isupper(), but doesn't ignore punctuation.
	"""
	for j in x:
		if not j.isupper():
			return False
	return True

text = [word for word in text if not (isallupper(word) or
	isallupper(word.rstrip('.')) or isnum(word))]

# Mess with RDDs

# text2 is text_rdd, but with all words "pushed down" one, so ['a','b','c']->['b','c','a']
# text3 has words "pushed down" two, so ['a','b','c']->['c','a','b']
text2 = [text[-1]]+text[:-1]
text3 = text[-2:]+text[:-2]

text_rdd = sc.parallelize(text)
text2_rdd = sc.parallelize(text2)
text3_rdd = sc.parallelize(text3)
temp = text2_rdd.zip(text_rdd)
temp2 = text3_rdd.zip(temp)
count = temp2.map(lambda v: ((v[0],v[1][0]),[(v[1][1],1)]))

# Remove lines that began with words from the end of the text file

count = count.filter(lambda x: x[0][0] not in ['END:', '***'])
count = count.reduceByKey(lambda x,y: x+y)

def get_counts(x):
	newlist = []
	found = []
	for i in x:
		if not i in found:
			newlist.append((i[0], x.count(i)))
			found.append(i)
	return newlist

count = count.map(lambda (k,v): (k,get_counts(v)))

# Sanity check

assert count.lookup(('Now','is')) == [[('the', 9),
  ('be', 1),
  ('it', 3),
  ('Mortimer', 1),
  ('that', 1),
  ('your', 1),
  ('his', 1),
  ('this', 1),
  ('he', 1),
  ('a', 1),
  ('my', 2)]]
  
# Generate sentences

def sample(arr1, arr2): # arr1 is list of appearance counts, arr2 is list of words
	"""
	Draws a random sample from arr2 weighted by the values in arr1.
	Example: arr1=[1,2,3], arr2=['a','b','c']. Then there is a 1/6 chance of returning
	'a', a 1/3 chance of 'b', and a 1/2 chance of 'c'.
	"""
	r = rand.randint(1,sum(arr1)+1)
	summed = 0
	for i in enumerate(arr1):
		summed += i[1]
		if summed >= r:
			return arr2[i[0]]

sentence_list = []
for i in range(10): # number of sentences
	start = count.keys().takeSample(True, 1)[0]
	sentence = ''
	sentence += start[0] + ' ' + start[1]
	for i in range(18): # length of sentence-2
		next_list = count.lookup(start)
		words, appears = zip(*next_list[0])
		next_word = sample(appears, words)
		start = (start[1],next_word)
		sentence += ' '+next_word
		
	sentence_list.append(sentence)
	
for i in sentence_list:
	print i, '\n'