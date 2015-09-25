# Your code here
import pyspark
sc = pyspark.SparkContext(appName="Spark1")
import numpy as np 
import itertools


# returns all orderings of a string
# these are not necessarily all valid words
def get_anagrams(s):
	return ["".join(perm) for perm in itertools.permutations(s)]

wlist = sc.textFile('EOWL_words.txt')

sorted_letter_to_word = wlist.map(lambda x: (''.join(sorted(x)), x)).groupByKey().map(lambda (x, y): (x, len(y), list(y))).sortBy(lambda x: (-1)*x[1])
print sorted_letter_to_word.first()
#words_per_letter_seq = sorted_letter_pairs

#print wlist.take(10)
#sorted_letters = wlist.map(lambda x: ''.join(sorted(x)))
#print sorted_letters.take(10)
#sorted_letters_anagrams = sorted_letters.map(lambda x: (x, get_anagrams(x)))
#all_anagrams = sorted_letters_anagrams.flatMapValues(lambda x: x)
#reverse_pairs = all_anagrams.map(lambda (x, y): (y, x))
#print all_anagrams.take(10)
#words = wlist.map(lambda x: (x,0))
#print reverse_pairs.take(10)
#print words.take(10)
#joined = words.join(reverse_pairs)
#joined.take(10)

# On the left is a word, on the right is a sorted
#remove_excess = joined.mapValues(lambda (x, y): y)







#xs = sc.parallelize(range(2000))
#grid = xs.cartesian(xs)
#grid.take(10)
#xys = grid.map(lambda (i, j): ((i, j), (j/500.0 - 2, i/500.0 - 2))).repartition(100)
#reses = xys.mapValues(lambda (x, y): mandelbrot(x, y))
#print xys.take(10)
#print reses.take(10)
#draw_image(reses)
#distribution = sum_values_for_partitions(reses).collect()
#plt.figure()
#plt.hist(distribution)
#plt.show()
