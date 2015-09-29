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
    if word == '':
        return False
    return True


if __name__ == "__main__":
    N = 16  # partitions

    sc = pyspark.SparkContext("local[4]")
    text_rdd = sc.textFile('shakespeare.txt', N, False)

    text_rdd = text_rdd.flatMap(lambda sentence: sentence.split(' '))
    text_rdd = text_rdd.filter(check_word)

    # build word lists with index
    first_rdd = text_rdd.zipWithIndex().map(lambda (k, v): (v, k))
    second_rdd = first_rdd.map(lambda (k, v): (k - 1, v))
    third_rdd = second_rdd.map(lambda (k, v): (k - 1, v))

    # join and drop index
    word_combo_rdd = first_rdd.join(second_rdd, N).join(third_rdd, N).map(lambda (i, ((w1, w2), w3)): ((w1, w2, w3), 1))
    word_combo_rdd = word_combo_rdd.reduceByKey(lambda v1, v2: v1 + v2)

    # redefine keys and group by word pairs
    result = word_combo_rdd.map(lambda ((w1, w2, w3), n): ((w1, w2), (w3, n))).groupByKey().mapValues(lambda l: list(l))

    print result.takeOrdered(10, key=lambda (k, v): -len(v))
