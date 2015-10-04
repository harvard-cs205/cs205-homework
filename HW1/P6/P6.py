import pyspark
from collections import Counter
from numpy.random import choice

def loadAndFilter(filename, sc):
    words = sc.textFile(filename).flatMap(lambda l: l.split())

    def wordFilter(word):
        return len(word) and not word.isdigit() and not word.isupper() and not (word[:-1].isupper() and word[-1] == '.')
    
    words = words.filter(wordFilter)

    # (i+2, word)
    indexWordRdd = words.zipWithIndex().map(lambda (w, i): (i+2, w))
    # (i+1, word), (i+2, word.next)
    indexWordRddShiftOne = indexWordRdd.map(lambda (i, w): (i-1, w))
    # (i, word), (i+1, word.next), (i+2, word.next.next)
    indexWordRddShiftTwo = indexWordRddShiftOne.map(lambda (i, w): (i-1, w))
    # ((word, word.next), word.next.next)
    combinedThreeWords = indexWordRdd.join(indexWordRddShiftOne).join(indexWordRddShiftTwo).values()
    # (((word, word.next), word.next.next), 1)
    combinedThreeWordsWithCount = combinedThreeWords.map(lambda kv: (kv, 1))
    # (((word, word.next), word.next.next), N)
    combinedThreeWordsWithCount = combinedThreeWordsWithCount.reduceByKey(lambda c1, c2: c1 + c2)
    # ((word, word.next), [(word.next.next, N)])
    combinedThreeWordsWithCount = combinedThreeWordsWithCount.map(lambda ((k, v), c): (k, [(v, c)]))

    markovWords = combinedThreeWordsWithCount.reduceByKey(lambda l1, l2: l1 + l2).map(lambda kv: kv)

    return markovWords

if __name__ == '__main__':
    sc = pyspark.SparkContext(appName='YK-P6')

    filename = 'ssby.txt'
    markovRdd = loadAndFilter(filename, sc)
    markovRdd.cache()

    #print markovRdd.keys().collect()
    #print markovRdd.map(lambda kv: kv).lookup(('Now', 'is'))
    
    phrases = []
    sources = markovRdd.takeSample(False, 10)
    for src in sources:
        srcKey = src[0]
        phrase = [srcKey[0], srcKey[1]]
        for i in xrange(18):
            entry = markovRdd.lookup(srcKey)[0]
            entryWords, entryProbs = [t[0] for t in entry], [float(t[1]) for t in entry]
            if len(entryWords) == 0:
                break
            totalProb = sum(entryProbs)
            entryProbs = [f / totalProb for f in entryProbs]
            newWord = choice(entryWords, p=entryProbs)
            phrase.append(newWord)
            srcKey = (srcKey[1], newWord)
        phrases.append(' '.join(phrase))
    
    for p in phrases:
        print p
    