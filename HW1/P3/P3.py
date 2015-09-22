import findspark; findspark.init()
import pyspark
import numpy as np

if __name__ == "__main__":
    N = 16  # partitions
    sc = pyspark.SparkContext("local[4]")  # number of workers
    wlist = sc.textFile('words.txt').cache()
    wlist.partitionBy(N, lambda word: np.ceil(np.random.uniform(0, N)))  # hash function discovered in P2
    word_word = wlist.map(lambda word: (''.join(sorted(word)), [word]), True)
    similar_words = word_word.reduceByKey(lambda l1, l2: l1 + l2)
    anagrams = similar_words.map(lambda entry: (entry[0], len(entry[1]), entry[1]), True)
    sorted_anagrams = anagrams.sortBy(lambda entry: entry[1], False, N)
    # result
    print sorted_anagrams.take(1)[0]
