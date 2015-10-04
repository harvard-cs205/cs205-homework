from pyspark import SparkContext, SparkConf

#Setup
conf = SparkConf().setAppName("anagrams").setMaster("local[*]")
sc = SparkContext(conf=conf)

#For each word, create a key out of its ordered sequence of letters
words = sc.textFile('EOWL_words.txt')
sortedLetters_words = words.map(lambda w: (tuple(sorted(w)), [w]))

#For each key, accumulate and count all words with that key
#Here + is list concatenation
anagrams = sortedLetters_words.reduceByKey(lambda x, y: x + y)
anagramCounts = anagrams.map(lambda w: (w[0], len(w[1]), w[1]))

#Retrieve the word with the most anagrams
result = anagramCounts.takeOrdered(1, lambda tup: -tup[1])
with open('P3.txt', 'w') as f:
    f.write("{}".format(result))
print result
