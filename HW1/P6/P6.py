import findspark
findspark.init('/Users/Grace/spark-1.5.0-bin-hadoop2.6/')
import pyspark
import re

sc = pyspark.SparkContext()

def filter_words(word):
    num_format = re.compile("^[0-9]+$")
    capital_foramt = re.compile('^[A-Z]+$')
    capital_with_period = re.compile('^[A-Z]+\.$')

    isnumber = re.match(num_format, word)
    iscapital = re.match(capital_foramt, word)
    iscapital_period = re.match(capital_with_period, word)

    if isnumber:
        return False
    elif iscapital:
        return False
    elif iscapital_period:
        return False
    else:
        return True



raw_data = sc.textFile("Shakespeare.txt")
#print str(raw_data.take(1)[0])
parsed_words = raw_data.flatMap(lambda x : x.split())
#print parsed_words.take(100)
#print parsed_words.count()
filtered_words = parsed_words.filter(filter_words)
#print filtered_words.take(20)
#print filtered_words.count()
words_index = filtered_words.zipWithIndex() #(word, index)
#print words_index1.take(10)
words_index1 = words_index.map(lambda (word, index) : (index, word))
words_index2 = words_index.map(lambda (word, index) : (index-1, word))
#print words_index2.take(10)
words_index3 = words_index.map(lambda (word, index) : (index-2, word))
#print words_index3.take(10)
joined_words = words_index1.join(words_index2).join(words_index3) #(index, ((w1, w2), w3))
joined_words.cache()
#print joined_words.sortByKey().take(10)
#print joined_words.count()
flatten_join = joined_words.map(lambda (index, ((w1, w2), w3)) : ((w1, w2, w3), 1)) #((w1, w2, w3), 1)
#print flatten_join.take(10)
sum_words = flatten_join.reduceByKey(lambda x, y : x + y) #((w1, w2, w3), counts)
#print sum_words.count()
detach_word3 = sum_words.map(lambda ((w1, w2, w3), count) : ((w1, w2), [(w3, count)]))
shakespeare_rdd = detach_word3.reduceByKey(lambda x, y : x + y)
#print shakespeare_rdd.take(10)
shakespeare_rdd.cache()

#print shakespeare_rdd.takeSample(True, 1)
#test = shakespeare_rdd.takeSample(True, 1)[0][1]
#print weighted_random_choice(test)
#print test

import random
RANDOM_PHRASES = 10

def weighted_random_choice(word3_list): #[(w3a, c3a), (w3b, c3b), ...]
    my_list = []
    for (w, c) in word3_list:
        my_list = my_list + [w]*c

    return random.choice(my_list)

def generate_text(rdd, number_of_words):
    first_two_words_tuple = rdd.takeSample(True, 1)
    text = []

    word1 = first_two_words_tuple[0][0][0]
    word2 = first_two_words_tuple[0][0][1]
    text.append(word1)
    text.append(word2)
    word3_list = first_two_words_tuple[0][1]

    while len(text) < number_of_words:
        word3 = weighted_random_choice(word3_list)
        text.append(word3)
        word3_list = rdd.map(lambda x: x).lookup((text[-2], text[-1]))[0] #get last 2 words and use them as key (w1, w2)

    return ' '.join(text)



for i in range(1, RANDOM_PHRASES+1):
    print i, ".", generate_text(shakespeare_rdd, 20)

