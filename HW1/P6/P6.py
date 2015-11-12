import re
import random
def main():
	#Text Parsing
	parsed_text=(full_text
             .map(lambda x: re.sub(r"\d+", "", x))
             .map(lambda x: re.sub(r"[A-Z]+\.", "", x))
             .map(lambda x: re.sub(r"[A-Z]+$|[A-Z]+[^a-zA-Z0-9_\']", "", x)))
	words=(parsed_text.flatMap(lambda x:re.split(' ',x)).map(lambda x:x.strip(".,:;?-][!()'"))
       .filter(lambda x: x != '')
       .zipWithIndex()
       .filter(lambda x:x[1]>773)
       .map(lambda x: x[0]))
	words_array=words.collect()
	#Markov chain creation
	indices=sc.range(words.count()-2)
	triples=indices.map(lambda x:((words_array[x],words_array[x+1]),words_array[x+2]))
	markov_words=triples.groupByKey().mapValues(lambda words:[(word,list(words).count(word)) for word in set(words)])
	# Sentence creation
	starting_words = markov_words.takeSample(False,10)
	sentences=[]
	for starter in starting_words:
	    sentence=list(starter[0])
	    for i in range(2,20):
	        print i
	        possible_words=markov_words.map(lambda x:x).lookup((sentence[i-2],sentence[i-1]))[0]
	        if len(possible_words)==1:
	            sentence.append(possible_words[0][0])
	        else:
	            total=sum(count for word,count in possible_words)
	            rand=random.uniform(0,total)
	            threshold = 0
	            for word, count in possible_words:
	                if threshold + count > rand:
	                    sentence.append(word)
	                    break
	                threshold += count
	    sentences.append(sentence)