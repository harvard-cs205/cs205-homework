import findspark
findspark.init()
import pyspark
import random


if __name__ == "__main__":  
    sc = pyspark.SparkContext()

    # Load the lines of the text file
    shak = sc.textFile("100.txt")

    # Split the lines into individual words
    words = shak.flatMap(lambda line: line.split())

    # Filter out the number-only words
    no_nums = words.filter(lambda word: not word.isnumeric())

    # Filter out words that contain only capital letters
    # This step also filters out words that contain only capital letters and end with a period
    no_allCaps = no_nums.filter(lambda word: word.upper() != word)

    # Take the words out of an RDD into a list to create the 3 word combos. 
    wordlist = no_allCaps.collect()

    # Create 3 lists that will line up to make 1st, 2nd, 3rd word groups.  
    wl1 = wordlist[:-2]  # slice off the last two entries, which will not be the start of 3 word groups
    wl2 = wordlist[1:-1] # slice off first and last entries
    wl3 = wordlist[2:]   # slice off first two entries
    
    # Put the lists into RDDs and zip them with indices, then switch the order of the keys and values by 
    #     mapping a lambda function so that they can be used as keys in tuple operations
    wl1RDD = sc.parallelize(wl1).zipWithIndex().map(lambda x: (x[1], x[0]))
    wl2RDD = sc.parallelize(wl2).zipWithIndex().map(lambda x: (x[1], x[0]))
    wl3RDD = sc.parallelize(wl3).zipWithIndex().map(lambda x: (x[1], x[0]))
    
    # Join the first, second and third word sets by joining based on key value.  The sequential joins give 
    #    every three word set in the format ((word1, word2), word3)
    wordSets = wl1RDD.join(wl2RDD).join(wl3RDD)

    # Take just the values, ie get rid of the indices
    wordSetVals = wordSets.values()

    # Define a function to map to grouped.  For all the third words that come after a two word key, take the 
    #    unique words, and return tuples of each word and the number of times it occurs.  
    def func(c): return [(i, list(c).count(i)) for i in (set(c))]

    # Map the above function to the grouped word pairs.  Add no-effect map to deal with the lookup bug
    sets = wordSetVals.groupByKey().mapValues(func).map(lambda x: x)

    # Verify the test case from the assignment
    #print sets.lookup(("Now", "is"))
    
    shakSentences = range(10)
    for y in range(10):

        # Empty set to which random words will be appended
        sents = []

        # Sample a pair of words and remove the outer brackets.  
        l0 = sets.takeSample(True, 1)[0]
    
        # Assign the first two words to the list, and instantiate the last word, which is used in the loop.  
        sents.append(l0[0][0])
        lastword = l0[0][1]
        sents.append(lastword)

        # Instantiate thirdwords, which is the set of potential third words with their weights.  
        thirdwords = l0[1]

        for x in range(18):
            # Create an empty sample space of potential third words to sample from
            samplespace = []

            for i in thirdwords: 
	        for z in range(i[1]):
	            samplespace.append(i[0]) # Append each word as many times as its weight to the sample space
        
            # Select one of the words from the weighted list of potenial words
            nextWord = random.sample(samplespace, 1)[0] 
        
            # Append this selected word to the list
            sents.append(nextWord)

            nextPair = (lastword, nextWord)
            lastword = nextWord

            # Look up the next pair and remove the outer brackets
            thirdwords = sets.lookup(nextPair)[0]

        shakSentences[y] = ' '.join(sents)

    for t in range(len(shakSentences)): print shakSentences[t]    
    
    
