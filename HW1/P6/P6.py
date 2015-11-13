import findspark
findspark.init()

import pyspark
import csv
import re 
import numpy as np
import time 

def gen_phrase(all_trip_counts):
    # now we have to take a sample of the code rdd 
    # should we pick the first words 
    (w_1, w_2), next_w_lst = all_trip_counts.takeSample(True, 1)[0]
    #print next_w_lst
    # then we will continue 
    cur_words = (w_1, w_2) 
    phrase = w_1 + " " + w_2   
    for i in range(18): 
        # split up the list into words, counts
        words, counts = zip(*all_trip_counts.lookup(cur_words)[0])
        probs = np.array(counts)
        probs = probs/float(probs.sum())
        next_word = np.random.choice(words, p=probs)
        phrase += " " + next_word
        cur_words = (cur_words[1], next_word)
    return phrase

def main():
    sc = pyspark.SparkContext()
    sc.setLogLevel("WARN")
    # reg ex to match the disallowed patterns
    reg_ex = r"(?:\A[0-9]+\Z)|(?:\A[A-Z]+\Z)|(?:\A[A-Z]+\.\Z)"
    words = sc.textFile('pg100.txt').flatMap(lambda x: x.split()).filter(
                                         lambda x: not bool(re.match(reg_ex, x))).collect()

    word_trips = sc.parallelize([((words[i], words[i+1], words[i+2]), 1) 
                                                for i in range(len(words)-2)])

    trip_counts = word_trips.reduceByKey(lambda x, y: x + y).map(
                            lambda (trip, count): ((trip[0], trip[1]), [(trip[2], count)]))

    all_trip_counts = trip_counts.reduceByKey(lambda x, y: x + y).cache()
    
    for i in range(10):
        print "Phrase %d" % i, gen_phrase(all_trip_counts)

if __name__=="__main__":
    main()
