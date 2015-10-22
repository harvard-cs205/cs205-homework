import findspark
findspark.init('/home/shenjeffrey/spark/')
import pyspark
import matplotlib.pyplot as plt
import seaborn as sns

# initiate spark
sc = pyspark.SparkContext()

# Read in data
data = sc.textFile("EOWL_words.txt")
data.take(20)

# Sort every word
# key = sorted word and value is original word
data = data.map(lambda word: (''.join(sorted(word)), word))

print data.take(15)

# Group the data by Key to produce a list of words that match with the key
result = data.groupByKey()

# Map in the length of word list
result = result.map(lambda(key, val_list): (key, len(val_list), val_list))
print result.take(10)

# Sort based on len(val_list) descendingly
anagrams_sorted = result.sortBy(lambda x: x[1], False)

# Print final results
print anagrams_sorted.take(1)[0]

# Print final results
for key, num, val_list in anagrams_sorted.take(1):
    print "key: ", key
    print "number of anagrams: ", num
    for word in val_list:
        print "word: ", word
