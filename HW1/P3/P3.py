import findspark
findspark.find()
findspark.init('/usr/local/opt/apache-spark/libexec')
import pyspark

sc = pyspark.SparkContext()

text_file = open("EOWL_words.txt", "r")
lines = text_file.readlines()
text_file.close()
lines = [i[0:-1] for i in lines]

# wlist = sc.textFile("EOWL_words.txt")
wlist = sc.parallelize(lines, 20)

#Return RDD in the form (SortedLetterSequence, NumberOfValidAnagrams)
sequence_num_RDD = wlist.map(lambda word: (''.join(sorted(list(word))),1)).reduceByKey(lambda x,y: x+y)

#Return RDD in the form (SortedLetterSequence1, [Word1a, Word2a, ...])
sequence_words_RDD = wlist.map(lambda word: (''.join(sorted(list(word))),''+ word)).groupByKey().map(lambda x : (x[0], list(x[1])))

#Join the two RDDs above and map to designated format
result = sequence_num_RDD.join(sequence_words_RDD).map(lambda x: (x[0], x[1][0], x[1][1]))

#Extract and print the line from the RDD above with the largest number of valid anagrams.
result = result.sortBy(lambda x: x[1], False)

print result.first()