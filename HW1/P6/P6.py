from pyspark import SparkConf, SparkContext
from collections import Counter
import numpy as np

def isValidWord(word):
    # the filter function
    return not word.isupper() and not word.isdigit() and not (word[:-1].isupper() and word[-1] == '.')

def MarkovPhrase(summaryrdd, len):
    # how we generate random phrase
    summaryrdd.cache()
    phrase = []
    phrase += list(summaryrdd.takeSample(True, 1)[0][0])

    for jj in range(20 - 2):
        try:
            nextlist = summaryrdd.lookup((phrase[-2], phrase[-1]))[0]
            phrase.append(np.random.choice(list(Counter(dict(nextlist)).elements())))
        except:
            break
    return ' '.join(phrase)

if __name__ == '__main__':

    conf = SparkConf().setAppName('KaiSquare')
    sc = SparkContext(conf = conf)
    
    text = sc.textFile('Shakespeare.txt')  # get the file
    words = text.flatMap(lambda line: line.split()).filter(isValidWord)  # split the line into words, and filter the invalid words
    indexedwords = words.zipWithIndex()  # zip with index, and then we can build triplets based on the indices
    indexed = indexedwords.map(lambda kv: (kv[1], kv[0]))
    indexedplus = indexed.map(lambda kv: (kv[0]+1, kv[1]))
    indexedplusplus = indexed.map(lambda kv: (kv[0]+2, kv[1]))

    coupletlist = indexedplusplus.join(indexedplus)  # join by shifted index
    tripletlist = coupletlist.join(indexed)  # join again and get ((a, b), c) format
    tripletlist = tripletlist.map(lambda kv: kv[1])

    print tripletlist.take(5)
    
    summary = tripletlist.groupByKey().mapValues(list)
    summary = summary.map(lambda kv: (kv[0], Counter(kv[1]).most_common()))
    summary = summary.sortByKey()

    print summary.lookup(('Now', 'is'))  # the target pair

    for ii in range(10):
        print MarkovPhrase(summary, 20)
