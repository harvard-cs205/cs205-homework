import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import pyspark
sc = pyspark.SparkContext()
sc.setLogLevel("ERROR")

wlist = sc.textFile("EOWL_words.txt")

#Create tuples with key being the alphabetically sorted string corresponding to the word saved in value:
w_tuple_string = wlist.map(lambda w: (list(w.lower().encode("utf-8")),w.encode("utf-8")))
w_tuple_string = w_tuple_string.map(lambda (key,value) : (''.join(sorted(key)),[value]))

#Grouping by key in order to get all anagrams :
result = w_tuple_string.reduceByKey(lambda v1,v2: v1+v2)

#Adding the number of anagrams for each tuple:
result = result.map(lambda (key,value): (key,len(value),value))

#Printing the top 10 anagrams:
print result.takeOrdered(10,lambda a: -a[1])