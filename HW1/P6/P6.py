import findspark
findspark.init('/home/shenjeffrey/spark/')
import pyspark
%matplotlib inline
import matplotlib.pyplot as plt

# initiate spark
sc = pyspark.SparkContext()

# Read in data
data = sc.textFile("shakespeare.txt")
data.take(10)

data = data.flatMap(lambda x: x.split(" "))

def filterNonWords(word):
    # take out empty strings
    if word == '':
        return False
    # take out words w/ only numbers
    elif word.isdigit():
        return False
    # take out words with all Capital characters
    # will also take out words with Capital characters w/ a period at the end
    elif word == word.upper():
        return False
    else:
        return True

clean_data = data.filter(filterNonWords)
clean_data2 = clean_data.collect()[1:] + [""]
clean_data3 = clean_data2[1:] + [""]

# Put them back into RDD
clean_data2 = sc.parallelize(clean_data2)
clean_data3 = sc.parallelize(clean_data3)

# Use Zip to create the key
clean_data = clean_data.zipWithIndex().map(lambda (x,y): (y,x))
clean_data2 = clean_data2.zipWithIndex().map(lambda (x,y): (y,x))
clean_data3 = clean_data3.zipWithIndex().map(lambda (x,y): (y,x))

# Creat n_gram
n_gram = clean_data.join(clean_data2).join(clean_data3)
n_gram = n_gram.map(lambda (key, val): ((val[0][0], val[0][1],val[1]), (1)))
n_gram = n_gram.reduceByKey(lambda x,y: x+y)
n_gram = n_gram.map(lambda (key, val): ((key[0],key[1]),(key[2], val)))
n_gram = n_gram.groupByKey().mapValues(list)

# Result
result = n_gram.collectAsMap()
result['Now', 'is']

def generate_weights(third_word_list):
    weights = []
    for word in third_word_list:
#         print word
        weights += [word[0]] * word[1]
    return weights

def generate_sentence(n_gram, sentence_length=20):
    # Declare blank sentence
    sentence = []
    # Take a random sample from the RDD
#     random_sample = n_gram.takeSample(True, 1,12)
    random_sample = n_gram.takeSample(True, 1)
    # First word
    first_word = random_sample[0][0][0]
    second_word = random_sample[0][0][1]
    third_word_list = random_sample[0][1]
    look_up_key = (first_word, second_word)
#     print "first word: ",first_word
#     print "second word: ",second_word
#     print "third word list: ",third_word_list
#     print "starting lookup: ",look_up_key
    sentence += [first_word] + [second_word]
#     print "start sentence: ",sentence
    # loop until sentence length is greater than 20
    while len(sentence) < sentence_length:
        # create weights
        weights = generate_weights(third_word_list)
#         print "weights: ", weights
        # use np.random.choice
        third_word = np.random.choice(weights)
#         print "third word: ", third_word
        
        # update sentence and look up key
        sentence += [third_word]
        look_up_key = (look_up_key[1], third_word)
#         print "look_up_key: ", look_up_key
        third_word_list = n_gram.map(lambda x: x).lookup(look_up_key)
#         print "new third word list: ", third_word_list
        third_word_list = third_word_list[0]
        
    print ' '.join(sentence)
    return ' '.join(sentence)

# Create sentences
for i in xrange(10):
    generate_sentence(n_gram, 20)