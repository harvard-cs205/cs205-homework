import numpy as np
import findspark
findspark.init('/home/toby/spark')

import pyspark

sc = pyspark.SparkContext(appName="Spark 2")
sc.setLogLevel('WARN') 

myfile = "pg100.txt"
lines = sc.textFile(myfile, 4)

def filter_words(line):
    words = line.split()
    def good_word(word):
        if word.isdigit(): return False
        if word.upper()==word: return False
        tmp = word[:-1]
        if len(word)>0 and word[-1]=="." and tmp.upper()==tmp: return False
        return True

    return [word for word in words if good_word(word)]

words = lines.flatMap(filter_words).collect()
print "making trigrams..."
trigrams = sc.parallelize([[(words[i], words[i+1], words[i+2]), 1.] for i in range(len(words)-2)])
trigrams = trigrams.reduceByKey(lambda a, b: a+b)
trigrams = trigrams.map(lambda KV: [(KV[0][0], KV[0][1]), [(KV[0][2], KV[1])]])
trigrams = trigrams.reduceByKey(lambda a, b: a+b)
trigrams = trigrams.partitionBy(16)

num_phrases = 10
num_words = 20
phrases = []
starts = trigrams.takeSample(True, num_phrases)

for start in starts:
    phrase = [start[0][0], start[0][1]]
    cur = start[1]

    for j in range(num_words-2):
        words = [word_counts[0] for word_counts in cur]
        prob = np.array([word_counts[1] for word_counts in cur])

        prob = prob/sum(prob)
        next_word = np.random.choice(words, 1, False, prob)[0]

        phrase.append(next_word)
        cur = trigrams.map(lambda x: x).lookup((phrase[-2], next_word))[0]

        print "phrase is ", phrase
        
        
    phrases.append(" ".join(phrase))
        
for phrase in phrases:
    print phrase
