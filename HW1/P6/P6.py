import numpy as np
from itertools import groupby


def is_not_number(s):
    try:
        float(s)
        return False
    except ValueError:
        return True

def not_all_caps(s):
	if s==s.upper():
		return False
	else:
		return True

def groupbyFreq(words):
	return dict((key, len(list(group))) for key, group in groupby(sorted(words)))

if __name__ == "__main__":
	words=sc.textFile("shakespeare.txt").flatMap(lambda i:i.split()) \
		.filter(is_not_number).filter(not_all_caps)
	
	#generate word group list of form (('w1','w2'),'w3')
	w1=words.zipWithIndex().map(lambda i:(i[1],i[0]))
	w2=w1.map(lambda i:(i[0]+1,i[1]))
	w3=w1.map(lambda i:(i[0]+2,i[1]))

	word_pairs=w3.join(w2).join(w1).values()
	
	phrase_freq=word_pairs.groupByKey()\
		.map(lambda i:(i[0],groupbyFreq(i[1])))
	phrase_list=word_pairs.groupByKey() \
		.map(lambda i:(i[0],list(i[1]))).cache() #we'll be using phrase_list a lot, 
													#so cache it

#Generate sentences
		sentences=''
	for x in range(10):
		pair1=phrase_list.takeSample(1,1)
		sen=list(pair1[0][0])
		sen_length=20
		for i in range(2,sen_length):
			sen.append(random.choice(phrase_list.lookup((sen[i-2],sen[i-1]))[0]))
		sentences=sentences+' '.join(sen)+'\n'
	with open("P6.txt", "w") as f:
		f.write(sentences)