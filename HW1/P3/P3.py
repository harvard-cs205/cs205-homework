# Initialize Spark
from pyspark import SparkContext
sc = SparkContext()
sc.setLogLevel("ERROR")

# Helper function - sorts the letters in a word alphabetically
# Returns sorted letters, original word

def sortWord(val):
    letters = [v for v in val]
    letters.sort()
    letters = ''.join(letters)
    return letters, val

# Load word list
wlist = sc.textFile('EOWL_words.txt')

# Use helper function to sort letters alphabetically
words_anagrams = wlist.map(lambda w: sortWord(w))

# Combine all words that contain the same letters
anagram_words_grouped = words_anagrams.groupByKey().mapValues(list)

# Count the number of anagrams for each word
anagram_words_count = anagram_words_grouped.map(lambda w: (w[0], len(w[1]), w[1]))

# Look up the word with the most anagrams
maxAnagrams = anagram_words_count.takeOrdered(1, lambda w: -w[1])

# Save the word with the most anagrams
text_file = open("P3.txt", "w")
text_file.write(str(maxAnagrams))
text_file.close()

# Also display result
print maxAnagrams