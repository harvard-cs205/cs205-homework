from pyspark import SparkContext
from urllib2 import urlopen
import os
import numpy as np

def parse_word(word):
    # only numbers 
    if word.isdigit():
        return False
    # only cap letters
    if word.isalpha() and word.isupper():
        return False
    # only cap letters that end with a period
    if word[:-1].isalpha() and word.isupper() and word[-1]=='.':
        return False
    return True

if __name__=='__main__':
    # initialize spark
    sc = SparkContext()

    # save text file
    textfile = 'Shakespeare.txt'
    if not os.path.exists(textfile):
        # download text file
        url = 'http://s3.amazonaws.com/Harvard-CS205/Shakespeare/Shakespeare.txt'
        with open(textfile, 'wb') as f:
            f.write(urlopen(url).read())

    # filter out words that contain only numbers 
    words = sc.textFile(textfile).flatMap(lambda line: line.split())
    words = words.filter(parse_word)

    # zip words with its element indices
    words = words.zipWithIndex()

    # create rdds with k:index/+1/+2 and v:word
    # if joined by key, words1 will have 1st word in a 3-word tuple
    words1 = words.map(lambda (word, i): (i+2, word))
    words2 = words.map(lambda (word, i): (i+1, word))
    words3 = words.map(lambda (word, i): (i, word))

    # join to get a list of 3-word tuples
    joined = words1.join(words2)
    joined = joined.join(words3)

    # remove index and count occurances of each distinct 3-word tuples
    joined = joined.map(lambda (i, ((w1, w2), w3)): ((w1, w2, w3), 1))
    counts = joined.reduceByKey(lambda x, y: x + y)

    # map to format ((word1, word2), [(word3, count)]) and aggregate by key
    counts = counts.map(lambda ((w1, w2, w3), c): ((w1, w2), [(w3, c)]))
    counts = counts.reduceByKey(lambda x, y: x + y)

    # generate 10 phrases each with 20 words
    startwords = counts.takeSample(True, 10)
    with open('P6.txt', 'wb') as f:
        for i in range(10):
            phrase = [None]*20
            phrase[0], phrase[1]= startwords[i][0]
            for j in range(2,20):
                wlist = counts.map(lambda x: x).lookup(
                    (phrase[j-2], phrase[j-1]))[0]
                wcounts = [c for w, c in wlist]
                prob = [float(c)/sum(wcounts) for c in wcounts]
                wlist = [w for w, c in wlist]
                phrase[j] = np.random.choice(wlist, p=prob)
            f.write('Phrase '+str(i+1)+'\n')
            f.write(' '.join(phrase) + '\n\n')
