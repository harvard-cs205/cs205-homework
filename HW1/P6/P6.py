from pyspark import SparkContext
from collections import Counter
import numpy as np

#initialize spark
sc = SparkContext('local', "Shakespeare")

#get the wordlist
wordlist = "/home/zhiqian/Dropbox/CS205/Homework/HW2_data/pg100.txt"



word_data = sc.textFile(wordlist)

word_data_split = word_data.map(lambda x: x.split(" "))
word_data_split2 = word_data_split.flatMap(lambda x: x)
print word_data_split2.take(10)
print word_data_split2.count()

#filter out the three conditions stated in the question
word_data_filtered = word_data_split2.filter(lambda x: (not str(x).isdigit()) 
											& (not all(map(str.isupper, str(x))))
											& (not (str(x).endswith('.') & all(map(str.isupper, str(x)[:-1]) )    )))

#give each word an index
word_data_first = word_data_filtered.zipWithIndex()
word_data_second = word_data_filtered.zipWithIndex()
word_data_third = word_data_filtered.zipWithIndex()

#change the index to the key, and stagger the index for second and third word
word_data_first_1 = word_data_first.map(lambda (x,y): (y,str(x)))
word_data_second_1 = word_data_second.map(lambda (x,y) : (y-1, str(x)))
word_data_third_1 = word_data_third.map(lambda (x,y) : (y-2, str(x)))

#filter out negative values
word_data_second_2 = word_data_second_1.filter(lambda (x,y): x >= 0)
word_data_third_2 = word_data_third_1.filter(lambda (x,y): x>= 0)

#join words based on index
first_two_words = word_data_first_1.join(word_data_second_2)
first_three_words = first_two_words.join(word_data_third_2)

#get rid of the index
first_three_words_2 = first_three_words.map(lambda (x, ((y,z),w)): ((y,z), w))

#using the first two words as key, groupbyKey
first_three_words_3 = first_three_words_2.groupByKey().mapValues(lambda x: list(x))

#count the elements in the list (third word)
first_three_words_4 = first_three_words_3.map(lambda ((x,y), z): ((x,y), Counter(z)))

key = (('Now','is'))
result = first_three_words_4.map(lambda x: x).lookup(key)
#result = first_three_words_4.filter(lambda ((x,y), z): (x == 'Now') & (y == 'is'))
print result


#----Now, to generate the Markov chain!-------------------------------

for counter in range(0,10):
	final_word_list = []
	#take a random sample. NOTE: i am working with the list of 3rd words BEFORE they are counted
	#as it is easier to take a random choice from there
	sample = first_three_words_3.takeSample(1, 1)
	#print sample
	final_word_list.append(sample[0][0][0])
	final_word_list.append(sample[0][0][1])
	#print list(sample[0][1])

	#randomly pick a 
	choice = np.random.choice(sample[0][1])
	#print choice
	final_word_list.append(choice)

	new_word = (sample[0][0][1], choice)

	#print new_word

	for x in range(3,20):
		#print "Running iterations:"+str(x) + " with " + str(new_word[0]) + " " + str(new_word[1])
		result2 = first_three_words_3.map(lambda x: x).lookup(new_word)
		#print result2

		#randomly pick a 
		choice = np.random.choice(result2[0])
		#print choice
		final_word_list.append(choice)

		new_word = (new_word[1], choice)

	print ' '.join(final_word_list)