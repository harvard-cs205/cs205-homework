import findspark
findspark.init('/Users/george/Documents/spark-1.5.0')
import pyspark
import random

sc = pyspark.SparkContext()

# filter out upper case and all digits
def filter_word(word):
    return not word.isdigit() and not word.isupper()

# generate sentence 
def create_phrase(rdd, word1, word2, length):
    sentence = [word1, word2]
    while len(sentence) < length:
        next_dist = rdd.lookup((word1, word2))[0]
        next_word = find_next_word(next_dist)
        sentence.append(next_word)
        
        word1 = word2
        word2 = next_word
    
    return " ".join(sentence)

# use a pdf for selecting the next word
def find_next_word(dist):
    total_count = sum([ele[1] for ele in dist])
    rand = random.randint(1, total_count)
    count = 0
    for word in dist:
        count += word[1]
        if count >= rand:
            return word[0]

# load data
lines = sc.textFile('Shakespeare.txt')

# flatten words and index by number
words = lines.flatMap(lambda x: x.split())
filtered = words.filter(filter_word)
indexed = filtered.zipWithIndex().map(lambda x: (x[1], x[0]))

# offset rdds by 1
offset_1 = indexed.map(lambda x: (x[0]+1, x[1]))
offset_2 = indexed.map(lambda x: (x[0]+2, x[1]))

# join and map rdd's to produce phrases of length 3
joined = offset_2.join(offset_1).join(indexed)
consec_3 = joined.map(lambda x: ((x[1][0][0], x[1][0][1]), [x[1][1]]))

# merge lists and produce format wanted by pset
reduced = consec_3.reduceByKey(lambda x,y: x+y)
answer = reduced.map(lambda x: (x[0], [(ele, x[1].count(ele)) for ele in list(set(x[1]))]))

# take 10 random samples
starts = answer.takeSample(False, 10)

# print phrases
for start in starts:
    word1 = start[0][0]
    word2 = start[0][1]
    print create_phrase(answer, word1, word2, 20)
    print
