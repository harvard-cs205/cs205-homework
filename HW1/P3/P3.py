import findspark 
findspark.init()
import pyspark

sc = pyspark.SparkContext()
sc.setLogLevel("WARN")

def main():
    # read in the words list
    wlist = sc.textFile('EOWL_words.txt')
    # turn into RDD of alphabetical order -> word
    alpha_words = wlist.map(lambda x: ("".join(sorted(x)), [x]))
    # RDD of alpha order -> [word1, word2, ...]
    alpha_anagrams = alpha_words.reduceByKey(lambda x, y: x + y)
    anagram_list = alpha_anagrams.map(lambda x: (x[0], len(x[1]), x[1]))
    max_anagram = anagram_list.sortBy(lambda x: x[1], ascending=False).take(1)
    print max_anagram

if __name__=="__main__":
    main()
