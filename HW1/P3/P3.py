import findspark
findspark.init('/Users/georgelok/spark')

import pyspark
sc = pyspark.SparkContext(appName="P3")

wlist = sc.textFile('EOWL_words.txt')

# Generate sorted anagram for each word as keys
anagramLists = wlist.map(lambda word: (''.join(sorted(word)), [word]))

# Merge anagrams together
mergedAnagramLists = anagramLists.reduceByKey(lambda x1, x2 : x1 + x2)

# Add counts to our RDD
mergedAnagramListsWithCounts = mergedAnagramLists.map(lambda (x,y) : (x, len(y), y))

# Sort in reverse by count.
finalList = mergedAnagramListsWithCounts.sortBy(lambda x : -x[1])

# Grab the anagram with the largest count
print finalList.first()