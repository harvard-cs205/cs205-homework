# anagram list maker

# initiallizing spark
from pyspark import SparkContext, SparkConf
sc = SparkContext(conf=SparkConf())
sc.setLogLevel("ERROR")

# get word list from file
# NOTE: copied text from given file into 'EOWL_words.txt'
wlist = sc.textFile('HW1/P3/EOWL_words.txt',use_unicode=False)

# make parallelized word list
words = sc.parallelize(wlist.collect())

# create dictionary of keys(alphabetized letters in word) and corresponding words
mydictionary = words.map(lambda x: (''.join(sorted(x)),x))

# group by words by key and put into list
mydictionary = mydictionary.groupByKey().mapValues(list)

# make it into correct format
mydictionary = mydictionary.map(lambda x: (x[0], len(x[1]), x[1]))

# to find the entry with maximum words
maxval = max(mydictionary.values().collect())
indexofmaximum = mydictionary.values().collect().index(maxval)
print mydictionary.collect()[indexofmaximum]
