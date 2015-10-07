
# coding: utf-8

# In[162]:

#Author: Xingchi Dai
#import the spark files
import findspark as fs;
fs.init();
import pyspark as py
from collections import Counter
import random
import re


#create sparkConf and a new Spark
#initialize a spark
conf = py.SparkConf().setAppName("CS205HW1")
sc = py.SparkContext();

#This function differentiate if the word is pure number or 
#if the word is all CAP and CAP + period
#Then the RDD will filter out these words
def filter_words(x):
    num_format = re.compile("^[0-9]+$");
    capital_foramt = re.compile('^[A-Z]+$');
    capital_with_period = re.compile('^[A-Z]+\.$');
    isnumber = re.match(num_format, x);
    iscapital = re.match(capital_foramt, x);
    iscapital_period = re.match(capital_with_period, x);
    return ((isnumber) and (iscapital) and (iscapital_period));

#The cut and the shift rdd functions are used for
#truncating extra elements in one list
#so we could union three lists into one
def cut(x):
    return (x[0] < counter - 2);
def shift_rdd(x):
    return (x[0] >= 0);

#The find_word function helps to find the third word
#The function will find the words that could be chosen from
#then the function use random function to decide which 
#word is chosen by the frequency of them(The choice is bias)
def find_word(x):
    #following is the algorithm to find the third number
    potential_words = counted_rdd.map(lambda x: x).lookup(x)[0];
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
    #return the last two words
    return_values=(result[-2], result[-1]);
    return return_values;


# In[120]:

#create a new rdd
rdd = sc.textFile("pg100.txt");
#split the words
words = rdd.flatMap(lambda x: x.split());
#filter the words
result = words.filter(filter_words).zipWithIndex().map(lambda (x, y) : (y, x));
#count the number of map
counter = len(result.collect());
#shift two times and filter out extra elements
original = result.filter(cut);
left_shift = result.map(lambda x:(x[0]-1,x[1])).filter(shift_rdd).filter(cut);
left_shift_two = result.map(lambda x:(x[0]-2,x[1])).filter(shift_rdd);
#combine three rdds to one
new_rdd = original.join(left_shift).join(left_shift_two).map(lambda x: x[1]);
#next we create the key/value pair then group done!
new_rdd = new_rdd.groupByKey();
#count all elements using collections library
counted_rdd = new_rdd.map(lambda x: (x[0], Counter(x[1]).most_common()));
counted_rdd.cache();


# In[172]:

#second part, genreate text from the model
#we need to run this script for 10 times
for script_time in range(10):
#take sample from the array
    sample = counted_rdd.takeSample(False,1)[0][0];
#look up the phrase, and find the most frequent one
    result = [];
    result.append(sample[0]);
    result.append(sample[1]);
    #using for loop to find those word
    inter = find_word(sample);
    # print "check the first inter"
    # print inter;
    for x in range(17):
        inter = find_word(inter);
    print "result is:";
    print script_time + 1;
    print ' '.join(result);

