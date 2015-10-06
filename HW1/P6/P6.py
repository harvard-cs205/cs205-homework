import numpy as np
import findspark
import pyspark

sc = pyspark.SparkContext()

def make_MarlovChains_inputdata(words):
    MC_rdd = sc.parallelize([[(words[i], words[i+1], words[i+2]), 1.] for i in range(len(words)-2)]).reduceByKey(lambda x,y: x+y)
    MC_rdd = MC_rdd.map(lambda xy: [(xy[0][0], xy[0][1]), [(xy[0][2], xy[1])]]).reduceByKey(lambda a, b: a+b).partitionBy(16)
    return MC_rdd

def create_new_sentences(num_s, num_w, MC_rdd):

    def bias(rdd_tmp):
        p_bias = np.array([x[1] for x in rdd_tmp])
        p_bias = p_bias/sum(p_bias)
        return p_bias

    sentences = []
    fs = MC_rdd.takeSample(True, num_s)

    for f in fs:
        comb = [f[0][0], f[0][1]]
        rdd_tmp = f[1]
        W=num_w-2

        for j in range(W):
            # find the next word biasedly by the counts  
            next_word = np.random.choice([x[0] for x in rdd_tmp] , 1, False, bias(rdd_tmp))[0]
            comb.append(next_word)

            rdd_tmp = MC_rdd.map(lambda x: x).lookup((comb[-2], next_word))[0]           
        sentences.append(" ".join(comb))
    return sentences


data = sc.textFile('pg100.txt')
#split words out into each word and remove all upper-case words
words = data.flatMap(lambda x: x.split()).filter(lambda x: x != x.upper())
words = words.filter(lambda x: not x.isdigit()).collect()
# make markov chains of order of 2
MC_rdd = make_MarlovChains_inputdata(words)
# create new sentences
sentences = create_new_sentences(10, 20, MC_rdd)
# output       
myfile = open('P6.txt', 'w')
for s in sentences:
    myfile.write(s)
    myfile.write('\n')
myfile.close()

