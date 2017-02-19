from pyspark import SparkContext

#initialize spark
sc = SparkContext('local', "Anagram")

#get the wordlist
wordlist = "/home/zhiqian/Dropbox/CS205/Homework/HW2_data/word_list.txt"

word_data = sc.textFile(wordlist)

#sort the sequence of each word and use it as the key
aggregator = word_data.map(lambda word: ((''.join(sorted(word))), word))

#group KV with the same Key, ie, they are anagrams
aggregator2 = aggregator.groupByKey()

#do a count on the KV
count_aggregator = aggregator2.map(lambda (x,y): (x, len(y), list(y)))

print count_aggregator.collect()

#output the largest count
print count_aggregator.takeOrdered(1, key = lambda (x,y,z): -y)