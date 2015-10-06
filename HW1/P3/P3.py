from pyspark import SparkContext

sc = SparkContext(appName="P3")
words = sc.textFile('wlist.txt')
counted_words = words.map(lambda x: (''.join(sorted(x)), [x])) \
															.reduceByKey(lambda words1, words2: words1 + words2) \
															.map(lambda (sorted_word, anagrams): (sorted_word, len(anagrams), anagrams)) \
															.sortBy(lambda (sorted_word, count, anagrams): count, ascending=False)

print counted_words.take(1)[0]