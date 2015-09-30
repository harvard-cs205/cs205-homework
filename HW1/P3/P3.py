# Initialize SC context
import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="Spark 1")

# Your code here
if __name__ == '__main__':
  # Set up Initial RDDs
  wlist = sc.textFile("EOWL_words.txt",100)
  rdd = wlist.map(lambda word: "".join(sorted(word)))
  # Find dictionary "sequence" : count
  count_result = rdd.countByValue()
  # Find dictionary "sequence" : list of anagrams
  list_of_result = wlist.groupBy(lambda word: "".join(sorted(word))).collectAsMap()
  # Map rdd to (SortedLetterSequence1, NumberOfValidAnagrams1, [Word1a, Word2a, ...])
  rdd_final = rdd.map(lambda seq: (seq,count_result[seq],list(list_of_result[seq])))
  
  print max(rdd_final.collect(),key=lambda item:item[1])
