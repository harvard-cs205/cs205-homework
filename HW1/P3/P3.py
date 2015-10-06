import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="P3")

#read a list of text files from a directory
text_files = sc.wholeTextFiles("EOWL-v1.1.2/LF Delimited Format/")

#load the words from each text file
def load_words(text_file):
	text_file_name, text_file_value = text_file	
	return text_file_value.split()

#combine all the words into a single list
words = text_files.flatMap(load_words)

#converts the list of words into a list of (sorted(word), word) tuples
words = words.map(lambda x: ("".join(sorted(x)), x))

#gets all words with the same sorted alphabetical key
anagrams = words.groupByKey()

#adds anagram count information
anagrams_and_counts = anagrams.map(lambda (x, anagrams_list): (x, len(anagrams_list), list(anagrams_list)))

#gets the word with the most anagrams (which is why we sort by -1 * count, to get the largest count)
most_anagrams = anagrams_and_counts.takeOrdered(1, key=lambda (x, anagrams_count, anagrams_list): -1 * anagrams_count)[0]
print most_anagrams

with open("P3.txt", "wb") as outfile:
	#outfile.write([most_anagrams[0], most_anagrams[1], most_anagrams[2]])
	print >> outfile, (most_anagrams[0], most_anagrams[1], most_anagrams[2])
