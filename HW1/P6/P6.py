import numpy as np

# Initialize Spark
from pyspark import SparkContext
sc = SparkContext()
sc.setLogLevel("ERROR")

# Helper functions

# Splits lines and removes uppercase words and digits
def clean_lines(line):
    result = []
    for r in line.split():
        upper = r.upper()
        if r != upper and not (r.isdigit()):
            result.append(r)
    return result

# Returns the third word in a sequence
# Probabilities are based on the frequency of the third word
def getLastWord(wordList):
    words = [i[0] for i in wordList]
    weights = [i[1] for i in wordList]
    weights = [i/float(sum(weights)) for i in weights]
    return [np.random.choice(words, p=weights)]

# Returns a phrase with a specified number of words
def getPhrase(rdd, noWords):
    
    firstWord = rdd.takeSample(True, 1)
    phrase = list(firstWord[0][0]) + getLastWord(firstWord[0][1])
    
    while len(phrase) < noWords:
        newWords = rdd.map(lambda x: x).lookup(tuple(phrase[-2:]))[0]
        phrase = phrase + getLastWord(newWords)
    
    result = ' '.join(phrase)
    return result

# Load text from file
wlist = sc.textFile('shakespeare.txt')

# Clean text (using helper function)
words = wlist.flatMap(lambda line: clean_lines(line))

# Assign an index to each word
words_index = words.zipWithIndex()

# Create 3 RDDs, each with an index incremented by 1
words_index_1 = words_index.map(lambda w: (w[1], w[0]))
words_index_2 = words_index_1.map(lambda w: (w[0]+1, w[1]))
words_index_3 = words_index_2.map(lambda w: (w[0]+1, w[1]))

# Group the RDDs to get three-word sequences
words_grouped = words_index_3.join(words_index_2).join(words_index_1).map(lambda w: (w[1], 1))

# Count the number of times each three-word sequence occurs
words_grouped_count = words_grouped.reduceByKey(lambda x, y: x + y).map(lambda w: (w[0][0], (w[0][1], w[1])))

# Group by the first two words in the three-word sequence
words_grouped_result = words_grouped_count.groupByKey().mapValues(list)

# Define parameters of output text
noPhrases = 10
noWords = 20

# Generate noPhrases
text = ''
for i in range(noPhrases):
    text = text + getPhrase(words_grouped_result, noWords) + '\n'

# Save phrases
text_file = open("P6.txt", "w")
text_file.write(text)
text_file.close()

# Print phrases
print text