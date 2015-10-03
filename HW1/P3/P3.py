# P3 solution
wordfile = 'EOWL_words.txt'

from pyspark import SparkConf, SparkContext
conf = SparkConf().setAppName('KaiSquare')
sc = SparkContext(conf = conf)

if __name__ == '__main__':
    wordlist = sc.textFile(wordfile)
    
    letterize = lambda word: ''.join(sorted(list(word)))
    word_letter = wordlist.map(lambda w: (letterize(w), w))

    word_join = word_letter.groupByKey().mapValues(list)
    word_final = word_join.map(lambda w: (w[0], len(w[1]), w[1]))
    print word_final.takeOrdered(1, key = lambda w: -w[1])
