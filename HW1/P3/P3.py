import pyspark
import itertools

sc=pyspark.SparkContext()

words=sc.textFile('EOWL_words.txt')

pair_words=words.map(lambda w: (''.join(sorted(w)), w))
anag_rdd=pair_words.groupByKey().mapValues(list)
anag_tri_rdd=anag_rdd.map(lambda ar: (ar[0], len(ar[1]), ar[1]))
top_anag=anag_tri_rdd.takeOrdered(1, lambda atr: -atr[1])

