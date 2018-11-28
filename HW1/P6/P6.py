#Setup
import pyspark
from pyspark import SparkContext
sc = SparkContext("local[8]")
sc.setLogLevel("ERROR")
data = sc.textFile('data.txt')
import random

#split words out into each word and remove all upper-case words
words = data.flatMap(lambda x: x.split()).filter(lambda x: x != x.upper())

def is_number(x): #from stackoverflow...this is how to remove all int words
    try:
        int(x)
        return False
    except ValueError:
        return True

words = words.filter(is_number)
string = words.collect()

#initialize RDD of each triplet of words. Could have used zipwith index but
#list/array manipulation in python likley needed before calling zipwithindex
#so i determined the below was just as simple/used python just as much

trips = []
for i in range(len(string)-2):
    trips.append(((string[i],string[i+1]),(string[i+2])))
trips = sc.parallelize(trips)

#from stackoverflow for determining count of words occurances in a list
def count(l):
    return [(x,l.count(x)) for x in set(l)] 

RDD = trips.groupByKey().map(lambda x: (x[0],list(x[1])))

Shakespeare = RDD.mapValues(lambda x: count(x))

#define function that will pick a third word based on a pair of words
def bias(x):
    bias_array = []
    for i in x:
        j = i[1]
        for k in range(j):
            bias_array.append(i[0])      
    return random.choice(bias_array)
  
#define function that calls bias to fill out a phrase of N = 'words' words  
def markov(words):
    prediction = []
    starting_pair = Shakespeare.takeSample(False,1)[0][0]
    prediction.append(str(starting_pair[0] ))
    prediction.append(str(starting_pair[1] ))
    i=2
    pair = (starting_pair[0],starting_pair[1])
    while i < (words):
        get_options = Shakespeare.lookup(pair)
        next_word = bias(get_options[0])
        prediction.append(str(next_word ))
        pair = (pair[1],next_word)
        i+=1
    return prediction

Generate_Shakespeare23 = markov(20)
print " ".join(Generate_Shakespeare23)