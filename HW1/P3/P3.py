from pyspark import SparkContext as sc
sc = sc(appName="P3")

wlist = sc.textFile('EOWL_words.txt')

sortedlist = wlist.map(lambda x: (''.join(sorted(x)), (1, [x])))
countlist = sortedlist.reduceByKey(lambda x, y: (x[0] + y[0], x[1]+y[1]))
countlist = countlist.sortByKey(keyfunc=lambda x: sorted(x))
anagrams = countlist.map(lambda x: (x[0], x[1][0], x[1][1]))
print anagrams.takeOrdered(1, key=lambda x: -x[1])