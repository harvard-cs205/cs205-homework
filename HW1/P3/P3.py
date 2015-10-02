import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext()

wlist = sc.textFile("EOWL_words.txt")

sorted_wlist_key_values = wlist.map(lambda word: (''.join(sorted(word)), [word]))
sorted_wlist_reduced = sorted_wlist_key_values.reduceByKey(lambda x, y: x+y)

output = sorted_wlist_reduced.mapValues(lambda lst: (len(lst), lst))

def maxValue(x, y):
	if x[1][0] > y[1][0]:
		return x
	else:
		return y

highestValue = output.reduce(maxValue)
print highestValue