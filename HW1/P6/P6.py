import findspark
findspark.init()
import pyspark


# helper method to check word validity
def check_word(word):
    try:
        int(word)
        return False
    except ValueError:
        pass
    if word.upper() == word:
        return False
    if word == u'':
        return False
    return True


if __name__ == "__main__":
    N = 16  # partitions

    sc = pyspark.SparkContext("local[4]")
    text_rdd = sc.textFile('shakespeare.txt', N)

    text_rdd = text_rdd.flatMap(lambda sentence: sentence.split(' '), True)
    text_rdd = text_rdd.filter(check_word)

    # sequential word lists
    second_words = text_rdd.collect()[1:] + [u'']
    third_words = second_words[1:] + [u'']

    # build triplet columns wit index
    first_rdd = text_rdd.zipWithIndex().map(lambda (k, v): (v, k), True)
    second_rdd = sc.parallelize(second_words, N).zipWithIndex().map(lambda (k, v): (v, k), True)
    third_rdd = sc.parallelize(third_words, N).zipWithIndex().map(lambda (k, v): (v, k), True)

    # join and drop index
    word_combo_rdd = first_rdd.join(second_rdd, N).join(third_rdd, N).map(lambda (i, ((w1, w2), w3)): ((w1, w2, w3), 1), True)
    word_combo_rdd = word_combo_rdd.reduceByKey(lambda v1, v2: v1 + v2)  # <----- reduction not okay! -->

    print word_combo_rdd.takeOrdered(10, key=lambda (k, v): -v)

    # redefine keys and group by word pairs
    result = word_combo_rdd.map(lambda ((w1, w2, w3), n): ((w1, w2), (w3, n))).groupByKey().mapValues(lambda l: list(l))

    print result.takeOrdered(10, key=lambda (k, v): -len(v))
