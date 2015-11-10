# Initialize SC context
import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="Spark 1")
import numpy as np
from P6_func import *

# Your code here
if __name__ == '__main__':
  # Set up Initial RDDs
  Shake_data = sc.textFile("Shakespeare.txt",100)
  # FlatMap RDD to words
  Shake_words = Shake_data.flatMap(lambda line: line.split())

  # Filtering out words
  # 1) only numbers
  Filtered_RDD = Shake_words.filter(lambda word: not (word.isdigit()))
  # 2) only letters which are capitalized
  Filtered_RDD = Filtered_RDD.filter(lambda word: not (word.isupper()))
  # 3) only letters which are capitalized and end with a period
  Filtered_RDD = Filtered_RDD.filter(lambda word: not (word.isupper() and list(word)[-1] == "."))
  # Filtering Only letters!
  #Filtered_RDD = Filtered_RDD.filter(lambda word: word.isalpha())

  # Create RDD ((Word1, Word2), [(Word3a, Count3a), (Word3b, Count3b), ...])
  Zip_RDD = Filtered_RDD.zipWithIndex().map(lambda (word,index): (index,word))
  Zip_RDD_Left_Shift_by_1 = Zip_RDD.map(lambda (index,word): (index-1,word))
  Zip_RDD_Left_Shift_by_2 = Zip_RDD.map(lambda (index,word): (index-2,word))

  # Join Zipped RDDs to get RDD ((Word1, Word2), Word3)
  Join_RDD_1 = Zip_RDD.join(Zip_RDD_Left_Shift_by_1).sortBy(lambda (index, words): index)
  Join_RDD_final = Join_RDD_1.join(Zip_RDD_Left_Shift_by_2).sortBy(lambda (index, words): index)
  Join_RDD_final = Join_RDD_final.map(lambda (index, ((word1, word2),word3)): ((word1,word2),word3)) 

  # Check the Example
  #Join_RDD_final.map(lambda x : x).lookup(('Now', 'is'))

  # Reduce Joined RDDs to get RDD ((Word1, Word2), (Word3, Count3))
  Reduced_RDD = Join_RDD_final.map(lambda (((word1,word2),word3)): (((word1,word2),word3),1) ) 
  Reduced_RDD = Reduced_RDD.reduceByKey(lambda x,y: x+y)
  Reduced_RDD = Reduced_RDD.map(lambda (((word1,word2),word3), count3): ((word1,word2),(word3,count3)))

  # Check the Example
  #Reduced_RDD.map(lambda x : x).lookup(('Now', 'is'))

  # Now convert Reduced RDD into ((Word1, Word2), [(Word3a, Count3a), (Word3b, Count3b), ...])
  #Final_Data_RDD = Reduced_RDD.reduceByKey(lambda x,y: x,y)
  Final_Data_RDD = Reduced_RDD.groupByKey().map(lambda (Key, Value): (Key,list(Value)))
  Final_Data_RDD.cache()
  
  # Check the Final Example 
  Final_Data_RDD.map(lambda x : x).lookup(('Now', 'is'))

  ### generate 10 random phrases from the model, each with 20 words
  random_phrases = []
  for i in range(10):
    # Choose a Random (Word1, Word2)
    Word1_Word2 = Final_Data_RDD.takeSample(False,1)[0][0]
    phrases = Generate_New_Phrases(Word1_Word2, Final_Data_RDD)
    random_phrases.append(phrases)

  print random_phrases[0]
  print random_phrases[1]
  print random_phrases[2]
  print random_phrases[3]
  print random_phrases[4]
  print random_phrases[5]
  print random_phrases[6]
  print random_phrases[7]
  print random_phrases[8]
  print random_phrases[9]
