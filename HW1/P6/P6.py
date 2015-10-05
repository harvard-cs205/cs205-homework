# Author : George Lok
# P6.py

from numpy.random import choice
from random import randint

import findspark
findspark.init('/Users/georgelok/spark')

import pyspark
sc = pyspark.SparkContext(appName="P6")

lines = sc.textFile('pg100.txt')

words = lines.flatMap(lambda line : line.split())

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

# Generate all possible ((first, second), (third,1)) tuples
new_list =[]
for i in range(len(words_list) - 2) :
    el = ((words_list[i], words_list[i+1]), (words_list[i+2], 1))
    new_list.append(el)

# Group by (first,second)
chain = sc.parallelize(new_list).groupByKey().map(lambda (x, y) : (x,list(y)))

# Merge counts for all (third, 1) tuples 
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

# Zip with index so that we can parallelize our 10 starting positions
finalChainWithIndex = finalChain.zipWithIndex().map(lambda ((firstPair, choices), index) : (firstPair, (choices,index)))

numPairs = finalChainWithIndex.count()

# Choose our 10 starting pairs to generate phrases.
pairSelection = choice(numPairs, 10)

# The first phase chooses the first 3 words of the 10 sentences, starting with the finalChainWithIndex RDD
def generateFirstPhrase((firstPair, (choices,index))) :
    if not index in pairSelection :
        return ([], [])
    wordList = []
    wordList.append(firstPair[0])
    wordList.append(firstPair[1])

    bins = []
    for (choice, count) in choices :
        for j in range(count) :
            bins.append(choice)
    index = randint(0,len(bins) - 1)
    nextWord = bins[index]
    wordList.append(nextWord)
    nextPair = (wordList[len(wordList) - 2], nextWord)
    
    return (nextPair, wordList)

# Subsequent phases use the results of the previous phase to generate the remaining 17 words for all sentences.  
# This can use the finalChain RDD
def createNextPhaseFunction(prevResults) :
    ref = {}
    for prev in prevResults :
        if prev[0] in ref :
            ref[prev[0]].append(prev[1])
        else :
            ref[prev[0]] = [prev[1]]
            
    def generateNextPhase((firstPair, choices)) :
        if not firstPair in ref :
            return [([],[])]
        results = []

        # It is entirely possible for us to have two of the same pair to appear, so
        # we account for this case by using flatmap.
        # Note that means that this might not strictly be fully parallel in this section
        for wordList in ref[firstPair] :
            bins = []
            for (choice, count) in choices :
                for j in range(count) :
                    bins.append(choice)
            index = randint(0,len(bins) - 1)
            nextWord = bins[index]
            wordList.append(nextWord)
            nextPair = (wordList[len(wordList) - 2], nextWord)
            results.append((nextPair, wordList))
        return results
    
    return generateNextPhase

# Run first phase
results = finalChainWithIndex.map(generateFirstPhrase).filter(lambda x : len(x[0]) > 0).collect()

# Run subsequent phases
for i in range(17) :
    results = finalChain.flatMap(createNextPhaseFunction(results)).filter(lambda x : len(x[0]) > 0).collect()

for result in results :
    print " ".join(result[1]) + "\n"