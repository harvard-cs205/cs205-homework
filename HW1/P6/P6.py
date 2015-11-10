import pyspark
import re
import random

def splitLine(line):
    return line.split()

def filter_word(word):
    not_numbers = len(re.sub('[0-9]', "", word)) > 0
    not_caps_period = len(re.sub('[A-Z]+\.*', "", word)) > 0
    return not_numbers and not_caps_period

def format_rdd(node, words_list):
    index = node[1]
    if index < len(words_list) - 2:
        return ((node[0], words_list[index+1]), [(words_list[index+2], 1)])
    else:
        return None

def update_hash(word_hash, arr):
    for word, count in arr:
        if word in word_hash:
            word_hash[word] += count
        else:
            word_hash[word] = count
    
def reduce_tuples(x, y):
    word_hash = {}
    update_hash(word_hash, x)
    update_hash(word_hash, y)
    return word_hash.items()

def find_next(words):
    choice = random.randint(1, sum(map(lambda x: x[1], words)))
    count = 0
    for word in words:
        count += word[1]
        if count >= choice:
            return word[0]

def generate_phrase(num_words, rdd):
    phrase = []
    rand_sample = rdd.takeSample(True, 1)
    phrase.append(rand_sample[0][0][0])
    phrase.append(rand_sample[0][0][1])
    while len(phrase) < num_words:
        lookup = rdd.map(lambda x: x).lookup((phrase[-2], phrase[-1]))
        if len(lookup) > 0:
            phrase.append(find_next(lookup[0]))
        else:
            return " ".join(phrase)
    return " ".join(phrase)

if __name__ == '__main__':
    sc = pyspark.SparkContext(appName="P6")
    lines = sc.textFile('Shakespeare.txt')
    words = lines.flatMap(splitLine)
    filtered_words = words.filter(filter_word)
    indexed_words = filtered_words.zipWithIndex().map(lambda x: (x[1], x[0]))
    offset_words_1 = indexed_words.map(lambda x: (x[0]-1, x[1]))
    offset_words_2 = indexed_words.map(lambda x: (x[0]-2, x[1]))
    indexed_words = indexed_words.join(offset_words_1).join(offset_words_2).map(lambda x: ((x[1][0][0], x[1][0][1]), [(x[1][1],1)]))
    word_model = indexed_words.reduceByKey(reduce_tuples)
    word_model = word_model.sortByKey().cache()
    phrases = []
    for i in range(0, 10):
        phrases.append(generate_phrase(20, word_model))
    print phrases


