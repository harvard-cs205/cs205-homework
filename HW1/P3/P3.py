import pyspark

# initialize spark context
sc = pyspark.SparkContext("local[4]", "Spark1")

# create RDD of word list from text file
wlist = sc.textFile('EOWL_words.txt')

# associate sorted character list & counter with each word
slist = wlist.map(lambda w: (''.join(sorted(w)), (1, [w])))

# combine words with same sorted character list
anagrams = slist.reduceByKey(lambda (n1, wlst1), (n2, wlst2): (n1 + n2, wlst1 +wlst2))

# flatten (since reduceByKey wouldn't play nice with multiple values)
anagrams_flat = anagrams.map(lambda (a, (b,c)): (a,b,c))

# get character list with greatest number of anagrams
print anagrams_flat.max(key=(lambda a: a[1]))
