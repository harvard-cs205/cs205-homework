from pyspark import SparkContext
from itertools import permutations

def clean_up_permutations(seq):
    """ Takes a list of all possible permutations,
    assumed to be in the format output by itertools.permutations
    - i.e., a list of characters - and returns a list of non-duplicate
    strings.
    """

    return list(set([''.join(p) for p in permutations(seq)]))

def find_real_anagrams(list_of_poss):
    """ Takes a list of possible anagrams (i.e., sequences of
    jumbled letters) and returns the subset of that list which are
    actually words.
    """

    return filter(lambda x: x in words, list_of_poss)


if __name__ == "__main__":

    # Get a SparkContext object
    sc = SparkContext('local', 'anagrams')

    # Load in the words
    words = sc.textFile('EOWL_words.txt')

    # Alphabetize the words
    # This returns sorted lists
    list_words = words.map(sorted)

    # And this joins the "list words" into usual strings
    alph_words = listWords.map(lambda x: ''.join(x))

    # Now remove duplicates
    distinct_sequences = alph_words.distinct()

    # We want to see every *distinct* way we can rearrange these sequences
    #all_possible_anagrams = distinct_sequences.map(lambda x: (x, clean_up_permutations(x)))
    all_possible_anagrams = distinct_sequences.map(lambda x: clean_up_permutations(x))

    # get the product of these with every word
    all_anas_and_words = all_possible_anagrams.cartesian(words).map(lambda (x, y): (x, [y]))

    # Reduce to get all words with every possible list
    all_anas_and_words = all_anas_and_words.reduceByKey(lambda x, y: x + y))


    # Filter by those which are actually words
    real_anagrams = all_possible_anagrams.map(lambda (seq, perms): (seq, find_real_anagrams(perms)))

    # Now get this in the proper form
    real_anagrams = real_anagrams.map(lambda seq, anas: (seq, len(anas), anas))


