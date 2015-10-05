import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName='anagrams')

with open('EOWL_words.txt', 'r') as rfile:
	words = rfile.read().split('\n')

lex_order = lambda s: ''.join(sorted(list(s)))

words_rdd = sc.parallelize(words, 10)
lex_to_word = words_rdd.map(lambda w: (lex_order(w), w))
lex_to_wordlist = lex_to_word.groupByKey()#.mapValues(list)
result_rdd = lex_to_wordlist.map(lambda (lex, wordlist) : (lex, len(wordlist), wordlist))
max_entry = result_rdd.max(key = lambda (lex, n, wordlist): n)
#  convert resultIterable to list for printing
max_entry = (max_entry[0], max_entry[1], list(max_entry[2]))
print max_entry
