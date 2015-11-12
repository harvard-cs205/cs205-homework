import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

	

if __name__ == "__main__":
	words=sc.textFile("words.txt")
	SortedLetters=words.map(lambda a: (''.join(sorted(a)),a))
	anagrams=SortedLetters.groupByKey() \
		.map(lambda i:(i[0],len(list(i[1])),list(i[1]))) \
		.sortBy(lambda x: x[1],False)
		#using map to turn the list of anagrams into a real list and not a ResultIterable 
		#object. Using sortBy to sort, the False makes it in descending order.
	anagram1=str(anagrams.first())#prints word with the most anagrams
	with open("P3.txt", "w") as f:
		f.write(anagram1)
		