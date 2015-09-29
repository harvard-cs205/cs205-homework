__author__ = 'Grace'

import pyspark

sc = pyspark.SparkContext()
words = sc.textFile('EOWL_words.txt')
#print words.take(10)

list_rdd = words.map(lambda w: (w, [w]))
sorted_rdd = list_rdd.map(lambda (w, wlist): (''.join(sorted(w)), wlist))
unique_rdd = sorted_rdd.reduceByKey(lambda w1_list, w2_list: w1_list + w2_list)
answer_rdd = unique_rdd.map(lambda (sorted_word, wlist): (sorted_word, len(wlist), wlist))

print_rdd = answer_rdd.sortBy(lambda (sorted_word, num, wlist): num, ascending=False)
maximum = print_rdd.take(1)
print maximum[0]

f = open("P3.txt", 'w')
f.write(str(maximum[0]))
f.close()