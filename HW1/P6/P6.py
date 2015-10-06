import numpy
import random
import findspark
import re
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="P4")

f = open("pg100.txt")
raw_text = f.read()
output = raw_text.split()

def cleanUp(word):
    is_word = False
    try:
        val = int(word)
    except ValueError:
        is_word = True
    if is_word == False:
        return False
    if word.isupper():
        return False
    if word[-1] == ".":
        if word[::-1].isupper():
            return False
    return True

assert(cleanUp("lasjflks") == True)
assert(cleanUp("JonAh") == True)
assert(cleanUp("bllahsk28askldH") == True)
assert(cleanUp("alksdf.") == True)
assert(cleanUp("1234") == False)
assert(cleanUp("51235151") == False)
assert(cleanUp("AJSDFJKAK") == False)
assert(cleanUp("AJSDFJKAK.") == False)
assert(cleanUp("AJSDFAKSFIWEIEIEIEJKAK.") == False)

cleaned = filter(cleanUp, output)

trigrams = []
for i in range(len(cleaned) - 2):
    trigrams.append((cleaned[i], cleaned[i + 1], cleaned[i + 2]))

shakespeare = sc.parallelize(trigrams)
trigram_vals = shakespeare.map(lambda (w1, w2, w3): ((w1, w2), [(w3, 1)]))

# input is a seq of word, count pairs, all of which are 1
def reduceFn(x, y):
    if x[0] == y[0]:
        return [(x[0], x[1] + y[1])]
    else:
        return x + y

def valMapper(seq):
    seqdict = {}
    for pair in seq:
        seqdict[pair] = 0
    for pair in seq:
        seqdict[pair] += 1
    output = []
    for i in seqdict.keys():
        output.append((i[0], seqdict[i]))
    return output

trigram_vals = trigram_vals.reduceByKey(lambda a, b: a + b)
mapped = trigram_vals.mapValues(valMapper)
mapped = mapped.map(lambda x: x)

def draw_biased_sample(seq):
    seqsum = float(sum([x[1] for x in seq]))
    probs = [x[1]/seqsum for x in seq]
    y = random.random()
    for a in xrange(len(probs)):
        y -= probs[a]
        if y <= 0:
            break
    return seq[a][0]

random_phrases = []
for i in range(10):
    phrase = ""
    initial = mapped.takeSample(False, 1)
    phrase += initial[0][0][0] + " " + initial[0][0][1]
    for j in range(18):
        next_word = draw_biased_sample(initial[0][1])
        phrase += " " + next_word
        initial = [((initial[0][0][1], next_word), mapped.lookup((initial[0][0][1], next_word))[0])]
    random_phrases.append(phrase)
print random_phrases
