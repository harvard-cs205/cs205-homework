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

	word_pairs=w3.join(w2).join(w1).values().cache() #we'll be using word_pairs a lot, 
													#so cache it
	phrase_freq=word_pairs.groupByKey()\
		.map(lambda i:(i[0],groupbyFreq(i[1])))
	
	
	