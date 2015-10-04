import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()

import numpy as np
import random

def counter(list):
    
    d={}
    for i in list:       
        d[i] = list.count(i)
    return d

def build_model(text):
	parsed_text = text.flatMap(lambda x: x.split(" ")).filter(lambda i: not str.isdigit(i.encode('ascii','ignore'))).filter(lambda j: not j.isupper()).filter(lambda k: not k=='')

	parsed_text1 = parsed_text.zipWithIndex()
	parsed_text2 = parsed_text.zipWithIndex()
	parsed_text3 = parsed_text.zipWithIndex()

	parsed_text1 = parsed_text1.map(lambda x: (x[1],x[0]))
	parsed_text2 = parsed_text2.map(lambda x: (x[1]+1,x[0]))
	parsed_text3 = parsed_text3.map(lambda x: (x[1]+2,x[0]))

	joined_23 = parsed_text3.join(parsed_text2)
	joined123 = joined_23.join(parsed_text1)
	joined123_2 = joined123.values()

	joined123_3 = joined123_2.groupByKey().map(lambda x: (x[0], list(x[1])))


	joined123_4 = joined123_3.map(lambda x: (x[0], counter(x[1])))

	return joined123_4


def build_phrases(text_model, number):
    all_phrases = []
    for j in xrange(number):

        starting_words = text_model.takeSample(False, 1, int(np.random.uniform(0,100000,1)))
        phrase = []
        w1,w2 = starting_words[0][0]
        phrase.append(w1)
        phrase.append(w2)
        #We make a list with the possible third words weighted by their number of ocurrences
        new_list = sum([([k[0]]*k[1]) for k in starting_words[0][1].items()],[])
        phrase.append(np.random.choice(new_list))

        for i in xrange(17):

            starting_words = text_model.lookup((phrase[i+1],phrase[i+2]))
            new_list = sum([([k[0]]*k[1]) for k in starting_words[0].items()],[])
            phrase.append(np.random.choice(new_list))
        
        phrase = ' '.join(phrase)
        all_phrases.append(phrase)

    return all_phrases


if __name__ == '__main__':

	text = sc.textFile('./Shakespeare.txt')

	shakespeare_model = build_model(text)

	random_phrases = build_phrases(shakespeare_model, 10)

	print random_phrases

	np.savetxt('./Random_Shakespeare.txt', random_phrases)


