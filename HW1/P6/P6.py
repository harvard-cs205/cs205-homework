import findspark
findspark.init()
import pyspark
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
sc = pyspark.SparkContext(appName="P6")

if __name__ == "__main__":
    textRDD = sc.textFile('Shakespeare.txt')
    wordRDD = textRDD.flatMap(lambda x: x.split(' '))
    filteredRDD = wordRDD.filter(lambda x: x.isdigit() == False).\
            filter(lambda x: x.isupper() == False)
    filteredRDD = filteredRDD.filter(lambda x: x != u'')
    rdd1 = filteredRDD.zipWithIndex()
    rdd2 = rdd1.filter(lambda x: x[1] > 0).map(lambda x: x[0]).zipWithIndex()
    rdd3 = rdd1.filter(lambda x: x[1] > 1).map(lambda x: x[0]).zipWithIndex()
    rdd1 = rdd1.map(lambda x:(x[1],x[0]))
    rdd2 = rdd2.map(lambda x:(x[1],x[0]))
    rdd3 = rdd3.map(lambda x:(x[1],x[0]))
    join_rdd1 = rdd1.join(rdd2)
    join_rdd2 = join_rdd1.join(rdd3)
    join_rdd3 = join_rdd2.map(lambda x:x[1]).map(lambda x:(x,1))
    #pairedListRDD has tuples as keys and lists as values
    pairedListRDD = join_rdd3.reduceByKey(lambda x,y:x+y)\
                    .map(lambda x:(x[0][0],[(x[0][1],x[1])]))\
                    .reduceByKey(lambda x,y: x + y)
    #pairRDD has tuples as keys and tuples as values
    pairRdd = join_rdd3.reduceByKey(lambda x,y:x+y)\
                .map(lambda x:(x[0][0],(x[0][1],x[1])))
    pairRdd.cache()
    sentence_list = pairedListRDD.takeSample(False, 10, 1)
    final_list = []
    for i in range(0,10):
        output_list = []
        start_words = sentence_list[i][0]
        print start_words
        output_list.append(start_words[0])
        output_list.append(start_words[1])
        for j in range(0,18):
            prob_list = []
            tok = pairRdd.lookup(start_words)
            s = 0
            for k in tok:
                s = s + k[1]
            for k in range(0,len(tok)):
                p = float(tok[k][1])/s
                prob_list.append(p)
            xk = np.arange(len(tok))
            custm = stats.rv_discrete(name='custm', values=(xk,tuple(prob_list)))
            R = custm.rvs(size=1)
            output_list.append(tok[R[0]][0])
            start_words = (output_list[j+1],output_list[j+2])
        print output_list
        final_list.append(output_list)
    print final_list
