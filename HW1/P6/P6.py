# Start the cluster
import seaborn as sns
sns.set_context('poster', font_scale=1.25)
import findspark as fs
fs.init()
import pyspark as ps
import multiprocessing as mp
import numpy as np

### Setup ###
config = ps.SparkConf()
config = config.setMaster('local[' + str(2*mp.cpu_count()) + ']')
config = config.setAppName('marvel_solver')
sc = ps.SparkContext(conf=config)

# Read in the data

raw_text = sc.textFile('Shakespeare.txt')
def get_desired_words(x):
    list_of_words = x.split()
    words_to_return = []
    for cur_word in list_of_words:
        # Remove numbers and all caps...ignores puncutation which is good
        if (not cur_word.isdigit()) and (not cur_word.isupper()) and (len(cur_word) != 0):
            words_to_return.append(cur_word)
    return words_to_return

filtered_text = raw_text.flatMap(get_desired_words)

# Now generate the groups of words in order
original_words = filtered_text.zipWithIndex()

no_shift = original_words.map(lambda x: (x[1], (x[0], 0))) # 0 represents no shift
right_shift = no_shift.map(lambda x: (x[0] - 1, (x[1][0], 1)))
two_right_shift = no_shift.map(lambda x: (x[0] - 2, (x[1][0], 2)))

combined_rdd = sc.union([no_shift, right_shift, two_right_shift])
groups_of_words = combined_rdd.groupByKey()

key_vs_words = groups_of_words.map(lambda x: (x[0], list(x[1])))

# Make sure that the words were inserted in the correct order within the groups
def keep_order(x):
    list_of_words = x[1]
    order = [z[1] for z in list_of_words]
    correct_order = np.argsort(order)
    list_of_words = [list_of_words[z] for z in correct_order]
    return (x[0], tuple(list_of_words))

keys_in_order_vs_words = key_vs_words.map(keep_order)

# We now prepare to count each word
def prepare_for_reducing(x):
    keys = []
    for word_tuple in x[1]:
        keys.append(word_tuple[0])
    keys = tuple(keys)
    return ((keys), 1)

grouping_preparation = keys_in_order_vs_words.map(prepare_for_reducing)

counts_of_phrases = grouping_preparation.reduceByKey(lambda x, y: x+y)

# We only want phrases of length 3
filtered_phrases = counts_of_phrases.filter(lambda x: len(x[0]) == 3)

# We now create the desired key-value list
new_key_vs_wordcount = filtered_phrases.map(lambda x: ((x[0][0], x[0][1]), (x[0][2], x[1])))

# We group by key to find all potential next words!
shake_phrases_iterable = new_key_vs_wordcount.groupByKey()

# We now get our final data!
shake_phrases = shake_phrases_iterable.map(lambda x: (x[0], list(x[1])))
# We sort the phrases for fast lookup
shake_phrases_sorted = shake_phrases.sortByKey().cache()

# We now generate and print our phrases
def get_phrase(shake_phrases_rdd, num_words=20):
    start_phrase = shake_phrases_rdd.takeSample(False, 1)[0]

    sentence = ''

    cur_key = (start_phrase[0][0], start_phrase[0][1])
    sentence += cur_key[0] + ' ' + cur_key[1] + ' '

    words_and_weights = start_phrase[1]
    for i in range(num_words):
        words = [z[0] for z in words_and_weights]

        weights = np.array([z[1] for z in words_and_weights])
        weights = weights/float(np.sum(weights))
        chosen_word = np.random.choice(words, p=weights)
        sentence += chosen_word + ' '

        # Now get a new key/value pair
        cur_key = (cur_key[1], chosen_word)
        words_and_weights = shake_phrases_rdd.lookup(cur_key)[0]
    return sentence

phrase_list = []
for i in range(10):
    phrase_list.append(get_phrase(shake_phrases_sorted))

for cur_phrase in phrase_list:
    print cur_phrase
    print