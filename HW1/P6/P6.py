import time
import findspark
findspark.init('/home/zelong/spark')
import pyspark
import numpy as np

sc = pyspark.SparkContext()

########################################################################################
#Clean the data and construct the RDD
########################################################################################
def only_capt_letter(x):
#this function will return true if x contain only letters and all letters are in capital
	if x.isalpha():
		if x.isupper():
			return True

def capt_letter_period(x):
#return true if string x contains only capitalized letters and end with a period
	if len(x) > 1:
		if x[-1] == ".":
			if x[:-1].isalpha():
				if x[:-1].isupper():
					return True	

lines = sc.textFile('pg100.txt')
words = lines.flatMap(lambda x: x.split(' '))
words_no_number = words.filter(lambda x: not x.isdigit())

words_no_cap = words_no_number.filter(lambda x: not only_capt_letter(x))

words_filtered = words_no_cap.filter(lambda x: not capt_letter_period(x)).filter(lambda x: not x=='')
#########################################################################################
#Shift word RDD
#Generate RDD with items containing 3 consecutive words
zip_index = words_filtered.zipWithIndex()
no_shift = zip_index.map(lambda x: (x[1], ( 0, x[0])))
left_shift1 = no_shift.map(lambda x: (x[0]-1, (1,x[1][1])))
left_shift2 = left_shift1.map(lambda x: (x[0] - 1, (2, x[1][1])  ))
collection_rdd = sc.union([no_shift, left_shift1, left_shift2])
phrase_rdd = collection_rdd.groupByKey().map(lambda (x,y): (x,[i for i in y])).filter(lambda x: len(x[1]) == 3)

def arrange_word(x):
#x in the form of (index, list of (0, string))
    index = x[0]
    words = x[1]
    result = sorted(words, key = lambda x: x[0])
    return (index, result)
        
phrase_ordered_words = phrase_rdd.map(lambda x: arrange_word(x)).map(lambda x: (x[0], tuple([i[1] for i in x[1]])))
ordered_phrase = phrase_ordered_words.sortByKey(True)

triple_words = ordered_phrase.map(lambda x: (x[1], 1))
triple_count = triple_words.reduceByKey(lambda x,y : x + y)
two_word_count = triple_count.map(lambda x: (tuple(list(x[0])[0:2]), (list(x[0])[2], x[1]) ))
pattern = two_word_count.groupByKey().map(lambda x: (x[0], [i for i in x[1]]))

##########################################################################################
#Function that based on two consecutive words, choose the thrid word
#Porbability will be higher on the words with more frequence

def generate_words(n):
    start = pattern.takeSample(True, 1)[0]
    result = list(start[0])
    for i in range(n-2):
        word = np.array([i[0] for i in start[1]])
        word_weight = np.array([i[1] for i in start[1]])
        total = 0.0
        for j in start[1]:
            total = total + j[1]
        word_weight = word_weight / total
        next_word = np.random.choice(word, p = word_weight)
        key = (start[0][1], next_word)
        result = result + [next_word]
        value = pattern.map(lambda x:x).lookup(key)[0]
        start = (key, value)
    return " ".join(result)

f = open('P6.txt', "w")

for i in range(10):
    sentence = str(i+1)+" : " + generate_words(20) + "\n"
    f.write(sentence)

f.close()

