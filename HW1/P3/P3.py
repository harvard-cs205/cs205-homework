import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext()

# make pyspark shut up
sc.setLogLevel('WARN')

wlist = sc.textFile("EOWL_words.txt")

# for each word, generate sorted characters as key and list containing the original word as value
sorted_wlist_key_values = wlist.map(lambda word: (''.join(sorted(word)), [word]))
# stitch all list values for the same sorted key
sorted_wlist_reduced = sorted_wlist_key_values.reduceByKey(lambda x, y: x+y)

output = sorted_wlist_reduced.mapValues(lambda lst: (len(lst), lst)).map(lambda (u, v): (u, v[0], v[1]))

def maxValue(x, y):
	if x[1] > y[1]:
		return x
	else:
		return y

highestValue = output.reduce(maxValue)
print highestValue