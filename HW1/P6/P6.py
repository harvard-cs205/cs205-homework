from pyspark import SparkContext
import random

if __name__ == "__main__":
    
    def wordFilter(word):
        if (word.isdigit()):
            return False
        if (word.isalpha() and word.isupper()):
            return False
        if (word.endswith(".") and word[:-1].isalpha() and word[:-1].isupper()):
            return False
        return True
    
    # initialize SparkContext
    sc = SparkContext(appName="Markov Shakespeare")
    
    # read text file into lines
    lines = sc.textFile("Shakespeare.txt")
    
    # split lines into words and then filter based on criteria as specified in wordFilter
    wordList = lines.flatMap(lambda line: line.split()).filter(wordFilter)
    
    # index each word based on position, orient the tuple accordingly, and cache the RDD
    wordTup = wordList.zipWithIndex().map(lambda tup: (tup[1], tup[0])).cache()
    
    # return (index of first word, ((word1, word2), word3))
    wordMap = wordTup.join(wordTup.map(lambda tup: (tup[0] - 1, tup[1]))).join(wordTup.map(lambda tup: (tup[0] - 2, tup[1]))).sortByKey()
    
    # remove the index, resulting in ((word1, word2), word3)
    wordMap = wordMap.map(lambda tup: ((tup[1][0][0], tup[1][0][1], tup[1][1]), 1))
    
    # reduce the map by the (word1, word2) key, and group the word3 words together
    reducedWordMap = wordMap.reduceByKey(lambda a, b: a + b)
    
    # reformat the key/value pairs, then group by the (word1, word2) keys; finally, sort and cache the final word map
    finalWordMap = reducedWordMap.map(lambda tup: ((tup[0][0], tup[0][1]), (tup[0][2], tup[1]))).groupByKey().mapValues(list).sortByKey().cache()
    
    # validate the results of finalWordMap
    # print(finalWordMap.lookup(("Now","is")).collect())
    
    # broadcast the final word map
    map = sc.broadcast(finalWordMap.collect())
    
    # choose 10 starting pairs of words to begin
    phrases = sc.parallelize(random.choice(map.value) for i in xrange(10))
    
    # rearrange into list format
    phrases = phrases.map(lambda tup: [tup[0][0], tup[0][1]])
    
    # convert the broadcast map into a dictionary
    dct = dict(map.value)
    
    # define a function to account for increased chance of appearing if seen multiple times
    def thirdWord(tup):
        thirdWords = dct[tup]
        words = []
        for (x,c) in thirdWords:
            for i in xrange(c):
                words.append(x)
        return random.choice(words)
    
    # iterate to obtain 20 words total
    for i in xrange(18):
        phrases = phrases.map(lambda phrase: phrase + [thirdWord((phrase[-2],phrase[-1]))])
    
    # obtain full phrases as a list of strings
    p = phrases.map(lambda phrase: " ".join(phrase)).collect()
    
    # iterate through and print phrases
    for x in p:
        print x + "\n"
        
    sc.stop()
