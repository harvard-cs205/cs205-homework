import findspark 
findspark.init()
from pyspark import SparkContext

sc = SparkContext()

wordsfile = sc.textFile("wordlist.txt")

#Alphabetizes the list of words
words = wordsfile.map(lambda w: ("".join(sorted(w)), w))

#create a rdd of the sorted words, number of anagrams, and list of anagrams
anagrams = words.groupByKey().map(lambda x : (x[0], len(list(x[1])), list(x[1])))

#sorts the anagram list by number of anagrams in descending order and then takes the first value
#aka, the largest number of anagrams 
max = anagrams.sortBy((lambda x: x[1]), ascending=False).take(1)
print max








