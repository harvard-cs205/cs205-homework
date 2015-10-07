from P4_bfs import *
sc.setLogLevel('ERROR')

data = sc.textFile('source.csv')

# clean up info into key by comic
data = data.map(lambda info : info[1:len(info)-1])
data = data.map(lambda info : info.split("\",\""))
data = data.map(lambda (chars, comic) : (comic, chars))

# have comic with all characters in it
shared = data.groupByKey()

# get each character listed with paired characters by comic
pairs = data.join(shared).values().map(lambda (K,V) : (K,list(V)))

# condense pairs into one list item per character
pairs = pairs.reduceByKey(lambda x,y : set(x) | set(y)).cache()



roots = ["CAPTAIN AMERICA", "MISS THING/MARY", "ORWELL"]


for char in roots:
	print char + ": "+ str(bfs(char, pairs, sc))
