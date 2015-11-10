from pyspark import SparkContext


if __name__ == "__main__":
    
    # initialize SparkContext
    sc = SparkContext(appName="Anagram")
    
    # read EOWL text file into wordList
    wordList = sc.textFile("/Users/overlord/test/EOWL_words.txt")
    
    # separate each word into a sorted list of characters, then create a tuple with this and the original word
    charList = wordList.map(lambda word: (''.join(sorted(list(word))), word))
    
    # reduce the map by the sorted list of characters, and group the full words together
    reducedList = charList.groupByKey().mapValues(list).sortByKey()
    
    # create the desired tuple: (SortedLetterSequence, NumberOfValidAnagrams, [Word1, Word2, ...])
    finalList = reducedList.map(lambda tup: (tup[0], len(tup[1]), list(tup[1])))
    
    # take the three anagrams resulting in 11 dictionary words each
    final = finalList.takeOrdered(3, key = lambda x: -x[1])

    # print out each anagram containing 11 dictionary words 
    for tup in final: print(tup)
    
    sc.stop()
