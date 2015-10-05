# initialize code
import findspark
findspark.init('/home/lhoang/spark')

import pyspark
sc = pyspark.SparkContext(appName='Spark1')

# load file
wlist = sc.textFile('EOWL_words.txt')

# map list of words to (sortedLetterSequence, count)
wkeys = wlist.map(lambda w: (''.join(sorted(w)), 1))

# reduce to get (sortedLetterSequence, numberOfValidAnagrams)
wkeys_reduced = wkeys.reduceByKey(lambda a, b: a + b)

# map list of words to (sortedLetterSequence, word)
wvalues = wlist.map(lambda w: (''.join(sorted(w)), w))

# group by key to get (sortedLetterSequence, [word1, word2, ...])
wvalues_group = wvalues.groupByKey().mapValues(list)

# join list of (sortedLetterSequence, numberOfValidAnagrams)
# with list of (sortedLetterSequence, [word1, word2, ...])
rdd = wkeys_reduced.join(wvalues_group)

# print ordered by highest count list of
# (sortedLetterSequence, numberOfValidAnagrams, [word1, word2, ...])

result = rdd.takeOrdered(10, lambda kkv: -kkv[1][0])

print result
