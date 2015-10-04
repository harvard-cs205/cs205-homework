import findspark
findspark.init('/Users/georgelok/spark')

import pyspark
sc = pyspark.SparkContext(appName="P6")

lines = sc.textFile('pg100.txt')

words = lines.flatMap(lambda line : line.split())

words.take(10)

def filterWords(word) :
    if(word.isdigit()) :
        return False
    if(word.isalpha() and word.isupper()) :
        return False
    if(word[0:-1].isalpha() and 
       word[0:-1].isupper() and 
       word[len(word)-1] == '.') :
        return False
    return True

new_words = words.filter(filterWords)

words_list = new_words.collect()

new_list =[]
for i in range(len(words_list) - 2) :
    el = ((words_list[i], words_list[i+1]), (words_list[i+2], 1))
    new_list.append(el)
    
chain = sc.parallelize(new_list).groupByKey().map(lambda (x, y) : (x,list(y)))


def chainMap((pair,words)) :
    temp = {}
    for word in words :
        if word[0] in temp :
            temp[word[0]] += 1
        else :
            temp[word[0]] = 1
    newWords = []
    for key, value in temp.iteritems():
        newWords.append((key,value))
    return (pair, newWords)
    
finalChain = chain.map(chainMap)

from random import randint

def generatePhrase() :
    wordList = []
    samples = finalChain.takeSample(True, 1)
    firstPair, choices = samples[0]
    wordList.append(firstPair[0])
    wordList.append(firstPair[1])

    # Choose remaining 20 words
    for i in range(18) :
        bins = []
        for (choice, count) in choices :
            for j in range(count) :
                bins.append(choice)
        index = randint(0,len(bins) - 1)
        nextWord = bins[index]
        wordList.append(nextWord)
        nextPair = (wordList[i+1], nextWord)
        choicesList = finalChain.lookup(nextPair)
        assert(len(choicesList) == 1)
        choices = choicesList[0]

    return wordList

results = []
for i in range(10):
    results.append(generatePhrase())

for result in results :
    print " ".join(result) + "\n"