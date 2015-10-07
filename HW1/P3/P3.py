from itertools import permutations, combinations
sc.setLogLevel('ERROR')

words = sc.textFile('EOWL_words.txt')

# disassemble into alphabetized letters of the word and the word
letters_words = words.map(lambda word : (sorted(list(word)), word))

# make the key the alphabetized-by-letter word, join all
joinedW = letters_words.map(lambda x : (''.join(x[0]),[x[1]])).reduceByKey(lambda x,y : x+y)

# sort by amount of anagrams
sortedW = joinedW.sortBy(lambda x: len(x[1]), ascending = False).map(lambda x : (x[0],x[1]))

print sortedW.take(1)