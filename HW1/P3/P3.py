from pyspark import SparkContext
from urllib2 import urlopen

# initialize spark
sc = SparkContext()

# load text file
url = 'http://s3.amazonaws.com/Harvard-CS205/wordlist/EOWL_words.txt'
words = urlopen(url).read().splitlines()
data = sc.parallelize(words)

# use sorted word as key and save count and list of words as values
data = data.map(lambda word: (''.join(sorted(word)), (1, [word])))

# for each sorted word, count the number of anagrams and collect all anagrams
result = data.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
result = result.map(lambda d: (d[0], d[1][0], d[1][1]))
firstline = result.takeOrdered(1, key=lambda d: -d[1])[0]

# print the line with the largest number of anagrams
with open('P3.txt', 'wb') as f:
    f.write(str(firstline))