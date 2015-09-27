import findspark
findspark.init()
import pyspark
import json
sc = pyspark.SparkContext(appName="Spark1")




# This function produce the final result format
# input----word: the word in the .txt file
# ouput----a tuple in the required format
def resultmapper(word):
    anagrams = []
    for i in word[1]:
        anagrams.append(i)
    return (''.join(sorted(word[1][0])),len(word[1]),anagrams)


# This function maps the format for the raw RDD. Using the buildin
# hash function to produce the key.
# Key is the unique hash code for the string and Value is the list of the word

# input----word: the word in the .txt file
# ouput----a Key, Value tuple
def K_V_mapper(word):
	key = hash(''.join(sorted(word)))
	value = [word]
	return (key,value)

# This is the function maps to reduceByKey, add up all the anagrams in one list
def reduce_mapper(item_1, item_2):
	return item_1 + item_2

rawRDD = sc.textFile('EOWL_words.txt')

pairedRDD = rawRDD.map(K_V_mapper) # each item in rawRDD is something like (hashCode, word) 
reducedRDD = pairedRDD.reduceByKey(reduce_mapper) # each item in reducedRDD is in format (hashCode, [list of anagrams])
resultRDD = reducedRDD.map(resultmapper).sortBy(lambda x:-x[1]) # firstly put the items into correct form. Then sort 
# it in descending order

maxLength = resultRDD.take(1)[0][1]
maxAnagrams = resultRDD.filter(lambda x : x[1] == maxLength)

print maxAnagrams.collect()

answer = resultRDD.take(1)[0]
f = open('P3.txt', 'w')
f.write(str(answer))
f.close()