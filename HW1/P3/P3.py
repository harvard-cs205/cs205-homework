from pyspark import SparkContext

sc = SparkContext("local", "Anagrams")

wList = sc.textFile("EOWL_words.txt")

# sort the letters of each word to create a key
sort_let = wList.map(lambda w: (sorted(list(w)), w))
let_key = sort_let.map(lambda (l, w): ("".join(l), [w]))

# group words with same sorted letter sequences together
group_keys = let_key.reduceByKey(lambda x, y: x + y)

# format according to homework specification
num_vals = group_keys.map(lambda (l, w): (l, len(w), sorted(w)))

# sort descending by number of words that are anagrams of each other
sorted_let_seq = num_vals.sortBy(lambda (l, n, w): -n)

print sorted_let_seq.take(1)
