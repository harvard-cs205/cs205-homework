import findspark;

findspark.init()
import pyspark
import itertools
import numpy as np


def anagrams_for_word(word):
    """For word return tuple: (word, [unique anagrams])"""
    return word, list({''.join(p) for p in itertools.permutations(word)})


def all_anagrams(words_with_anagrams):
    """Returns all anagrams"""
    anagrams = words_with_anagrams.map(lambda tuple: tuple[1])  # Take anagram lists
    anagrams = anagrams.reduce(lambda l1, l2: l1 + l2)  # one big list
    return sorted(list(set(anagrams)))  # filter unique and sort anagrams


def build_result(anagrams, words_with_anagrams):
    """Builds RDD with entries:
        (SortedLetterSequence1, NumberOfValidAnagrams1, [Word1a, Word2a, ...])
        (SortedLetterSequence2, NumberOfValidAnagrams2, [Word2a, Word2b, ...])
        etc."""
    sorted_words_with_anagrams = words_with_anagrams.sortByKey().cache()
    swwa = sorted_words_with_anagrams.collect()
    anagrams_with_words = anagrams.map(lambda anagram: (anagram, [word_and_list[0] for word_and_list in swwa if
                                                                  anagram in word_and_list[1]]), True)
    ang_with_num_and_words = anagrams_with_words.map(lambda entry: (entry[0], len(entry[1]), entry[1]), True)  # count
    result = ang_with_num_and_words.sortBy(lambda entry: entry[1], ascending=False)  # sort on number
    return result


if __name__ == "__main__":
    sc = pyspark.SparkContext()
    wlist = sc.textFile('s3://Harvard-CS205/wordlist/EOWL_words.txt').cache()
    wlist.partitionBy(8, lambda word: np.ceil(np.random.uniform(0, 8)))
    wlist.map(lambda word: word.lower())
    print wlist.take(10)
    # wlist = sc.parallelize(['hello', 'lleho', 'test'])
    words_with_anagrams = wlist.map(anagrams_for_word)
    anagrams = all_anagrams(words_with_anagrams)
    anagrams_rdd = sc.parallelize(anagrams, 8)
    result = build_result(anagrams_rdd, words_with_anagrams)
    print result.take(10)

