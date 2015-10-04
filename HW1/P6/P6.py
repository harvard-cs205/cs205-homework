import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="")
import re
import random

## Note that to I have excluded the licensen information from the downloaded for text for better accuracy
## Pre-process the text to create a list of words without all caps, all caps with an ending period, and all nums
llist = sc.textFile('source.txt')
wlist = llist.flatMap(lambda x: x.split())
wlist_filtered = wlist.filter(lambda x: (not x[:-1].isupper()) if x[-1] == '.' else ((not x.isupper()) and (not x.isdigit())))

## With the cleaned word list, index, offset and then join the list(s) to create (w1,w2),[(w3,w3_count),...]
wlist_index = wlist_filtered.zipWithIndex().map(lambda x: (x[1], x[0]))
wlist_offset1 = wlist_index.map(lambda x: (x[0]-1, x[1]))
wlist_offset2 = wlist_offset1.map(lambda x: (x[0]-1, x[1]))
wlist_join = wlist_index.join(wlist_offset1).join(wlist_offset2)
word_set = wlist_join.map(lambda x: x[1]).groupByKey().map(lambda (k,v): (k, list(v))).map(lambda (k, v): (k, [(x, v.count(x)) for x in set(v)])).cache()

## According the the 2 previous words, randomly choose the 3rd word weighted by the word count in the "dictionary"
## This is done by randomly selecting an item from a list of [word] * word_count
## Then we print the 2 sample key + 18 generated words
def sentence(num_sentences, num_words, dictionary):
    for i in range(num_sentences):
        sample = dictionary.takeSample(True, 1)
        sentence = [w for w in sample[0][0]]
        for j in range(2,num_words):
            next_choice_raw = dictionary.map(lambda x: x).lookup((sentence[-2], sentence[-1]))
            next_word = random.choice([w for wl in [[x[0]]*x[1] for x in list(*next_choice_raw)] for w in wl])
            sentence.append(next_word)
            sentence2 = ' '.join(sentence) + '\n'
        print sentence2

sentence(10, 20, word_set)