
# coding: utf-8

# In[1]:

import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="Spark 2")

import numpy as np

# import matplotlib.pyplot as plt 
# import matplotlib.cm as cm
import time


# In[154]:

text = sc.textFile('../../DataSources/pg100.txt')


# In[155]:

# Remove the following type of strings:
# -only numbers
# -only capitals
# -only capitals + period at end
def only_num(check_str):
    return all([x.isdigit() for x in check_str])
def only_caps(check_str):
    return all([x.isupper() for x in check_str])
def only_caps_period(check_str):
    return (any([~x.isdigit() for x in check_str[:-1]]) 
            and (check_str[-1] == '.'))


# In[156]:

words_raw = text.flatMap(lambda x: x.split(' '))


# In[414]:

# Simply taking out the non-words will create some artificial sequences. 
# But maybe this is better than creating holes that will cause dead-ends.
text_filtered = words_raw.filter(lambda word: (not only_num(word) 
                                        and not only_caps(word) 
                                        and not only_caps_period(word)))

# text_filtered = words_raw.map(lambda word: 
#                               (word if (not only_num(word) 
#                                         and not only_caps(word) 
#                                         and not only_caps_period(word)) 
#                                else u'NaW'))


# In[417]:

# Create RDD with triplets of words
text_zip = text_filtered.zipWithIndex().map(lambda x: (x[1], x[0]))
text_zip1 = text_zip.map(lambda x: (x[0] + 1, x[1]))
text_zip2 = text_zip.map(lambda x: (x[0] + 2, [x[1]]))
text_triplet_filt = (text_zip.join(text_zip1)
                .join(text_zip2)
                .map(lambda x: x[1])
                .cache())


# In[418]:

# def rmv_missing(x):
#     return ((x[0][0] != u'NaW') 
#             and (x[0][1] != u'NaW') 
#             and (x[1][0] != u'NaW'))
# text_triplet_filt = text_triplet.filter(rmv_missing)


# In[420]:

# Reduce by key
text_triplet_redu = text_triplet_filt.reduceByKey(lambda a, b: a + b)


# In[421]:

# Gather all the counts of each possibility
def val_counts(redund_list):
    words, counts = np.unique(redund_list, return_counts=True)
    return [(words[word_i], counts[word_i]) for word_i in range(len(words))]

text_counts = text_triplet_redu.map(lambda x: (x[0], val_counts(x[1])))

# Strange hack recommended by pset
text_counts = text_counts.map(lambda x: x).cache() 


# In[432]:

### Done with the markov model, now generate random sequences ###
# Take random sample from RDD
n_phrases = 10
init_pair_list = [elem[0] for elem in text_counts.takeSample(False, n_phrases)]


# In[433]:

def pick_next_word(curr_choices):
    choice_len = len(curr_choices)
    probs = [1.0 * x[1] / choice_len for x in curr_choices]
    return curr_choices[np.random.choice(choice_len, 1, probs)[0]][0]


# In[434]:

phrase_len = 20
all_phrases = []
for phrase_i in range(n_phrases):
    init_words = init_pair_list[phrase_i]
    curr_words = []
    curr_words.append(init_words[0])
    curr_words.append(init_words[1])

    for word_i in range(phrase_len - 2):
        new_key = (curr_words[-2], curr_words[-1])

        # look up element in RDD
        curr_choices = text_counts.lookup(new_key)[0]

        # pick next word
        new_word = pick_next_word(curr_choices)

        # append it to curr_words
        curr_words.append(new_word)
    all_phrases.append(curr_words)


# In[439]:

# Save list of strings to file
def save_str_list(test_strs):
    with open('P6.txt', 'a') as f:
        for word_i, word in enumerate(test_strs):
            f.write(word)
            if word_i < len(test_strs) - 1:
                f.write(' ')
            else:
                f.write('.\n\n')
                
for phrase in all_phrases:
    save_str_list(phrase)

