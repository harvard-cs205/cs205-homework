#import findspark; findspark.init();
import pyspark as py
from collections import Counter
import random
#initiate spark
def filter_words(x):
	return not (x.isdigit() or x.isupper());
def cut(x):
	return (x[1] < counter - 2);
def shift_rdd(x):
	return (x[1] >= 0);

def find_word(x):
	#following is the algorithm to find the third number
	potential_words = counted_rdd.map(lambda x: x).lookup(sample)[0];
	#print potential_words;
	#Use random choice to generate a word
	#create a word list
	word=[];
	weight=[]
	words_list=[]
	for x in potential_words:
		word.append(x[0]);
		weight.append(x[1]);
	#build a word list
	for x in range(len(word)):
		for y in range(weight[x]):
			words_list.append(word[x]);
	# choose one word out of the word_list
	third_word = random.choice(words_list);
	#add the new word to the string
	result.append(third_word);
	#result = result + " " + third_word;
	#print result; successfully
	#return the last two words
	return (result[-1], result[-2]);
	# print "xxxx"
	# print (result[-1],result[-2]);

#create sparkConf and a new Spark
conf = py.SparkConf().setAppName("CS205HW1")
sc = py.SparkContext();
#create a new rdd
rdd = sc.textFile("pg100.txt");
#split the words
words = rdd.flatMap(lambda x: x.split(" "));
#filter the words
result = words.filter(filter_words).zipWithIndex();
#count the number of map
counter = len(result.collect());
#shift two times and filter out extra elements
original = result.filter(cut);
left_shift = result.map(lambda x:(x[0],(x[1]-1))).filter(shift_rdd).filter(cut).map(lambda x:x[0]);
left_shift_two = result.map(lambda x:(x[0],x[1]-2)).filter(shift_rdd).map(lambda x:x[0]);
#combine three rdds to one
pairs = original.map(lambda x: x[0]).zip(left_shift).zip(left_shift_two);
#next we create the key/value pair then group done!
new_rdd = pairs.groupByKey();
#count all elements using collections library
counted_rdd = new_rdd.map(lambda x: (x[0], Counter(x[1]).most_common()));

#second part, genreate text from the model
#take sample from the array
sample = counted_rdd.takeSample(False,1)[0][0];
#look up the phrase, and find the most frequent one
result = [];
#result += sample[0] + " " + sample[1] 
result.append(sample[0]);
result.append(sample[1]);
#using for loop to find those word
inter = find_word(sample);
# print "check the first inter"
# print inter;
for x in range(8):
	inter = find_word(inter);
	# print "check the inter";
	# print x;
	# print inter;
print "let's see what the result is:"
print result;
