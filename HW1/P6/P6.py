import findspark
findspark.init()
import pyspark
import numpy as np


def make_sentence(triplet_rdd):
    """Builds sentence from dataset randomly"""
    sentence = []
    # sentence_length = int(np.round(np.random.normal(24.46, 3.85)))  # Average sentence English length according to [1]
    sentence_length = 20
    sample = triplet_rdd.takeSample(True, 1)[0]  # Random first words
    two_words = sample[0]
    third_word_list = sample[1]
    sentence += [two_words[0].title()]  # Captalizes first letter
    sentence += [two_words[1]]
    while len(sentence) < sentence_length:
        # Get and store third word
        if len(third_word_list) > 1:
            weighted_third_word_list = reduce(lambda l, (word, n): l + ([word] * n), third_word_list, [])
            third_word = np.random.choice(weighted_third_word_list)
        else:
            third_word = third_word_list[0][0]
        sentence += [third_word]
        # Update step
        two_words = (two_words[1], third_word)
        third_word_list = triplet_rdd.map(lambda x: x).lookup(two_words)[0]
        if not third_word_list:
            break  # if no (two, three) combination is found, end sentence prematurely (very unlikely)
    sentence = ' '.join(sentence)
    if sentence[-1] != '.':
        sentence += '.'  # End sentence nicely with a dot.
    return sentence


def check_word(word):
    """Helper method to check word validity"""
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

    print text_rdd.lookup('lala')

    # build word lists with index
    first_rdd = text_rdd.zipWithIndex().map(lambda (k, v): (v, k))
    second_rdd = first_rdd.map(lambda (i, v): (i - 1, v))
    third_rdd = second_rdd.map(lambda (i, v): (i - 1, v))

    # join and drop index
    word_combo_rdd = first_rdd.join(second_rdd, N).join(third_rdd, N).map(lambda (i, ((w1, w2), w3)): ((w1, w2, w3), 1))
    word_combo_rdd = word_combo_rdd.reduceByKey(lambda v1, v2: v1 + v2)

    # redefine keys and group by word pairs
    result = word_combo_rdd.map(lambda ((w1, w2, w3), n): ((w1, w2), (w3, n))).groupByKey().mapValues(lambda l: list(l))

    sentences = []
    for i in xrange(10):
        sentences += [make_sentence(result)]

    for sentence in sentences:  # to make sure they show up after the PySpark output
        print sentence

# [1] http://hearle.nahoo.net/Academic/Maths/Sentence.html

