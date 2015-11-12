import random
import numpy as np


if __name__ == "__main__":
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