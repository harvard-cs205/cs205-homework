import pyspark
from operator import add

def words_we_want(w):
    dialog_marker = w[-1] == '.' and w[:-1].isupper()
    return (not w.isdigit()) and (not w.isupper()) and (not dialog_marker)

if __name__ = "__main__":
    # initialize spark context
    conf = pyspark.SparkConf().setAppName("P6").setMaster("local[*]")
    sc = pyspark.SparkContext(conf=conf)

    # get words from text file
    shakespeare = sc.textFile('Shakespeare.txt')
    words = shakespeare.flatMap(lambda l: l.split(' ')).filter(words_we_want).zipWithIndex.cache()

    # build Markov table
    # first, get all sets of 3 consecutive words
    words0 = words.map(lambda (k,v): (v,k)).partitionBy(8).cache()
    words1 = words0.mapValues(lambda v: v-1).map(lambda (k,v): (v,k)).partitionBy(8).cache()
    words2 = words0.mapValues(lambda v: v-2).map(lambda (k,v): (v,k)).partitionBy(8).cache()
    consec_words = words0.join(words1).join(words2).values().map(lambda x: (x,1))

    # then count how many of each
    consec_counts = consec_words.reduceByKey(add)

    # and group by first two words of triplet
    markov_table = consec_counts.map(lambda (((w1,w2),w3),c): ((w1,w2),(w3,c))).groupByKey().cache()

    # check that we've done this correctly
    print markov_table.lookup(('Now','is'))
