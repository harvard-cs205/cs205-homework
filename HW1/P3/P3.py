__author__ = 'xiaowen'
import findspark
findspark.init('/home/xiaowen/spark')
import pyspark

# build sparkcontext object
sc = pyspark.SparkContext(appName="P3")

# read data, convert it to RDD
wlist = sc.textFile('/home/xiaowen/cs205-homework/HW1/P3/EOWL_words.txt')

# (K,V1,V2) where K is word sorted letter seq, V1 is 1 and V2 is the original word
sorted_words = wlist.map(lambda w: (''.join(sorted(w)), w))

# count the number of words for each seq and combine the words into a list
result_rdd = sorted_words.groupByKey().map(lambda x: (x[0], len(list(x[1])), list(x[1])))

# find the entry with the most anagrams
max_value = max(result_rdd.map(lambda x: x[1]).collect())

# filter the result_add by max_value and convert it to list
print result_rdd.filter(lambda x: x[1] == max_value).collect()

