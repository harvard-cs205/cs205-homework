import pyspark
from pyspark import SparkContext

sc = SparkContext()

#load in the textfile as a wlist
wlist = sc.textFile('EOWL_words.txt')

#things we will later append to
blank_string = ''
blank_list = []

#go through the word list and then set the sorted words as one value and then
#the actual word in a list as the word. 
#don't think I need these appends on either side, just had them earlier and kept
#them because it clairified for me what I was doing. 
list_and_words = wlist.map(lambda x: (blank_string.join(sorted(x)), blank_list + [x]) )

#all words combined in a list by the sorted letters as above.
matchedSeq = list_and_words.reduceByKey(lambda x, y: x + y)

#include the length 
sequence_number_list = matchedSeq.map(lambda (x, y): (x, len(y), y))

#sort so we can pick the longest one to write out. 
sorted_list = sequence_number_list.sortBy(lambda (x, y, z): y, ascending=False)


f = open('P3.txt', 'w')
f.write(str(sorted_list.take(1)))
f.close()







