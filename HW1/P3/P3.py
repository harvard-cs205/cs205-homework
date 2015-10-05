import csv
import os
from pyspark import SparkContext
sc = SparkContext()

# Insert a word into the collection
def insert(s):
	try:
		sorteds = sortString(s)
		if sorteds in hashmap:
			hashmap[sorteds] += [s]
		else:
			hashmap[sorteds] = [s]
	except:
		return
	return

# Counting sort. Using sort() / sorted() in python takes O(nlog(n)) times.
# counting sort takes linear time (O(n))
def sortString(s):
	charExist = [0 for i in xrange(26)]
	for elem in s:
		charExist[ord(elem)-97] += 1
	res = ""
	for i in xrange(26):
		for j in xrange(charExist[i]):
			res += chr(i+97)
	return res

# Create a tuple (sorted version of this word, this word itself)
def createTuple(s):
	sorteds = 0
	try:
		sorteds = sortString(s)
	except:
		sorteds = "#"
	return (sorteds, [s])

# Get the complete anagram list and output to a format
# (SortedLetterSequence1, NumberOfValidAnagrams1, [Word1a, Word2a, ...])
def getAnagramsParallel(path):
	wlist = []
	# Read in data from local
	for root, dirs, filenames in os.walk(path):
		for f in filenames:
			if f[-3:] == "csv":
				wlist += sc.textFile(path+f).collect()

	rdd = sc.parallelize(wlist, 50)
	rdd = rdd.map(createTuple)

	# Anagram RDD tuples with the second element as list of anagrams for a specific sorted key.
	anagramRDD = rdd.combineByKey(lambda x: x, lambda x, y: x+y, lambda x,y: x+y)

	# Add length to the tuple
	anagramRDD = anagramRDD.map(lambda x: (x[0], len(x[1]), x[1]))
	return anagramRDD


path = '/Users/wenshuaiye/CS205/cs205-homework/HW1/P3/EOWL-v1.1.2/CSV Format/'
rdd = getAnagramsParallel(path)
anagrams = rdd.filter(lambda x: x[1] == 11).collect()

# Output the entries with the most anagrams
with open("P3.txt", "w") as txtfile:
	for p in anagrams:
		out = ",".join(p[2])
		txtfile.write(out)
		txtfile.write("\n")
		txtfile.write("\n")




