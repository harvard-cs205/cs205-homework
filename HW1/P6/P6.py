import findspark
findspark.init()
import pyspark

# shut down the previous spark context
#sc.stop() 
sc = pyspark.SparkContext()
sc.setLogLevel('WARN')
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import pdb
import time
import pandas as pd
import itertools
import itertools as it

# Load the words from the file and extract them
lines = sc.textFile('Shakespeare.txt')
words = lines.flatMap(lambda line: line.split()).map(lambda x : x.encode('utf-8') ).map(lambda w: (w,1))

# Filter the digits from the text
words = words.filter(lambda w: False if w[0].isdigit() else True )

# Filter the words entirely capitalize or entirely capitalize and end with a period
words = words.filter(lambda w: False if w[0].isupper() else True ).keys()
words_index = words.zipWithIndex()
index_words = words_index.map(lambda (word, ind ): (ind,word) )

time_start = time.time()
final = []

for phrase_number in range(0,10):
    
    print 'phrase number: ', phrase_number
    print  'final: ', final
    
    if phrase_number==0:
        word1 = 'He' ; word2 = 'can'
    elif phrase_number==1:
        word1 = 'While' ; word2 = 'he'
    elif phrase_number==2:
        word1 = 'To' ; word2 = 'be'
    elif phrase_number==3:
        word1 = 'Because' ; word2 = 'he'
    elif phrase_number==4:
        word1 = 'If' ; word2 = 'it'
    elif phrase_number==5:
        word1 = 'For' ; word2 = 'me'
    elif phrase_number==6:
        word1 = 'We' ; word2 = 'are'
    elif phrase_number==7:
        word1 = 'You' ; word2 = 'are'
    elif phrase_number==8:
        word1 = 'They' ; word2 = 'are'
    elif phrase_number==9:
        word1 = 'There' ; word2 = 'is'
    
    number_words = 2
    phrase = [word1, word2]
    while number_words <= 19: 
        # create list from look up and find the (word, indexes) for word 1 and word 2
        where_word1  = words_index.lookup(word1)
        word1_rdd = words_index.filter( lambda w: w[1] in where_word1)

        where_word2 = words_index.lookup(word2)
        word2_rdd = words_index.filter( lambda w: w[1] in where_word2)

        # switch key and value of word1_rdd and word2_rdd and subtract 1 to index of word2_rdd
        word1_rdd = word1_rdd.map(lambda (k,v) : (v,k))
        word2_rdd = word2_rdd.map(lambda (k,v) : (v-1 ,k ))

        # Regroup by key since we want to check the index of word 1 when word 2 is after word 1
        indices = word1_rdd.join(word2_rdd).sortByKey()

        # Collect the rdd with indices for word 1 when followed by word 2 (this takes forever)
        indices_list = indices.keys().collect()
    

        # Filter the rdd with (index, word) from Shakespeare
        sentences = index_words.filter(lambda (ind3, w3): ind3-2 in indices_list).map(lambda (k,v):(v,k))
        sentences = sentences.groupByKey().map(lambda (k,v): (k, len(v) )).sortByKey()

        # Now take the most popular 3rd word
        word3 = sentences.takeOrdered(1, lambda x: -x[1])[0][0]
        number_words = number_words + 1
    
        # Construct the sentence
        phrase.append(word3)   
        print 'phrase', phrase
    

        # Now update
        word1 = word2
        word2 = word3
    
    final.append(phrase)
    phrase_number = phrase_number +1
    
time_stop = time.time()
elapse = time_stop - time_start
print elapse


# Write the 10 sentences in a text file
fd = open('writefinal.txt','w')
for phrase in final:
    for i in phrase:
        fd.writelines(i)
        fd.write(', ')
    fd.write('\n')
fd.close()