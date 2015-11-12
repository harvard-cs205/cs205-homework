############################################
### Problem 6 - Markov Shakespeare[25%]	 ###
### P6.py								 ###
### Patrick Day 						 ###
### CS 205 HW1                           ###
### Oct 4th, 2015						 ###
############################################

########################
### Import Functions ###
########################
import pyspark
import numpy as np
from operator import add

import os
sc = pyspark.SparkContext()
sc.setLogLevel('WARN')

#################
### Build RDD ###
#################
# Import Data and Split into Words
shake = sc.textFile('pg100.txt', use_unicode=False)
words = shake.flatMap(lambda x: x.split())

# Filter our spaces, uppercase, and digits
filter_words = words.filter(lambda x: x != " " and x != x.upper() and x != x.isdigit())

# Create RDD w/ index for "text steram" and adjecent words
text_stream_ind = filter_words.zipWithIndex()
txt_ind_1 = text_stream_ind.map(lambda x: (x[1], x[0]))

# Create the "shifted by 1" lists
shift_1 = filter_words.collect()[1:] 
shift_2 = shift_1[1:]

# Create RDDs for shifts
shift_1_rdd = sc.parallelize(shift_1, 10)
shift_2_rdd = sc.parallelize(shift_2, 10)

# Create "text stream" indexs for shifts
txt_ind_2 = shift_1_rdd.zipWithIndex().map(lambda x: (x[1], x[0]))
txt_ind_3 = shift_2_rdd.zipWithIndex().map(lambda x: (x[1], x[0]))

# Combined all text streams together
all_txt_combos = txt_ind_1.join(txt_ind_2).join(txt_ind_3)

# Create RDD according to the problem and reduce by key
predict_count = all_txt_combos.map(lambda (x,y): ((y[0][0], y[0][1], y[1]), 1))
predict_cnt_reduce = predict_count.reduceByKey(add)

# Combine Similiar 2-word combinations and list possiblities
two_word_key = predict_cnt_reduce.map(lambda (x,y): ((x[0], x[1]), (x[2], y)))
two_word_predict = two_word_key.groupByKey().mapValues(list).cache()

##########################
### Generate Sentences ###
##########################
# Sample from 2 word predictions & assign word fragments
sample_2_word = two_word_predict.takeSample(True, 1)
first_word = sample_2_word[0][0][0]; second_word = sample_2_word[0][0][1] 
predict_word = sample_2_word[0][1] 

# Initialize arrays prior to sentance builder
sent_builder = [first_word, second_word]
lookup_word_key = (first_word, second_word)
shakespeare_sent = []

### Build shakespeare_sent ###
for x in xrange(0, 10):
	# Select word based on occurances in next predicted word
    word_occur = [predict_word[0][0] for i in xrange(predict_word[0][1])]
    
    # Randomly select from all word choices and add to sentance
    new_word = np.random.choice(word_occur)
    sent_builder.append(new_word)
    
    # Create new lookup ket and find the new predicted word
    lookup_word_key = (lookup_word_key[1], new_word)
    predict_word = two_word_predict.map(lambda x: x).lookup(lookup_word_key)[0]
    
# Put it all together and print
shakespeare_sent = ' '.join(sent_builder)    
print("Shakespeare says \n", shakespeare_sent)

