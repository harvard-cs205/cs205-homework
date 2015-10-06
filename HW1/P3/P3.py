from pyspark import SparkContext

# Set up context and pull words
sc = SparkContext("local", "P3")
words = sc.textFile("EOWL_words.txt")

# Sort each word's letters and pair it with its original form
# (sorted, word)
sortedWords = words.map(lambda w: (''.join(sorted(w)), w))

# Group by key to find anagrams. Then introduce counts via a map
# (sorted, [word1, word2, ...], count)
groupedCountedWords = sortedWords.groupByKey().map(lambda (w, lst): (w, list(lst), len(lst)))

# Run a reduction to compute the max counted anagram
maxCount = groupedCountedWords.reduce(lambda tup1, tup2: max(tup1, tup2, key = lambda tup: tup[2]))

# Print the result
print "FINAL result: ", maxCount
