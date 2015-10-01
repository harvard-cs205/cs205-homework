import findspark
findspark.init('/home/zelong/spark')
import pyspark

sc = pyspark.SparkContext()

#generate desired RDD
word_rdd = sc.textFile("EOWL_words.txt")
anagram_rdd = word_rdd.map(lambda x: (''.join(sorted(x)), x) )
group_rdd = anagram_rdd.groupByKey()
list_rdd = group_rdd.map(lambda x: (x[0], [i for i in x[1]]) )
result_rdd = list_rdd.map(lambda x : (x[0], len(x[1]), x[1]))


#find the largest number of anagrams
#and write to tx file
def compare(x):
	return x[1]
max = result_rdd.top(3, key = compare)

f = open('P3.txt', "w")

for i in max:
    i = str(i)
    f.write(i+"\n")

f.close()


