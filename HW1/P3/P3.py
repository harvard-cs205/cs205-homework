from pyspark import SparkContext
import pdb

sc = SparkContext()
textFile = sc.textFile("EOWL_words.txt")
#Sort the letters of each word and make that key, and the original word the value
sortedTf = textFile.map(lambda word: (''.join(sorted(word)),word))
#Group words that are the same anagrams together with the count
v = sortedTf.groupByKey().map(lambda x: (len(list(x[1])),x[0],list(x[1]))).reduce(lambda a,b:max(a,b))

print v
