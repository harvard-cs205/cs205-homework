from pyspark import SparkContext

if __name__ == "__main__":

    # Get a SparkContext object
    sc = SparkContext('local', 'anagrams')

    # Load in the words
    words = sc.textFile('EOWL_words.txt')
    
    # Pair up all the words with themselves wrapped in a list
    pair_words = words.map(lambda x: (x, [x]))

    # Sort the first element as a key
    sorted_pair_words = pair_words.map(lambda (x, y): (''.join(sorted(x)), y))

    # Reduce by key and merge lists
    seqs_and_word_list = sorted_pair_words.reduceByKey(lambda x,y: x+y)

    # Now add in the number of words
    proper_format = seqs_and_word_list.map(lambda (x, y): (x, len(y), y))

    # And lets find the max
    max_finder = proper_format.map(lambda (x, y, z): (y, x, z)).max()

    with open("P3.txt", 'w') as out_file:
        print >> out_file, max_finder[1], max_finder[0], max_finder[2]
