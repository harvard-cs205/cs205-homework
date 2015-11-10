# Q3
wlist = sc.textFile('EOWL_words.txt')

maxAnagrams = wlist.map(lambda x: (''.join(sorted(x)), [x]))\
                    .reduceByKey(lambda x, y: x+y)\
                    .map( lambda (s, v): (s, len(v), v) )\
                    .reduce( lambda x, y: x if x[1] > y[1] else y )

print maxAnagrams
