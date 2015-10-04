source = sc.textFile("s3://Harvard-CS205/Shakespeare/Shakespeare.txt")
source = source.flatMap(lambda x : x.split())
from operator import add 
def validWord(word):
	if word.isdigit() or word.isupper() or (word[-1] == "." and word[:-1].isupper()):
		return False
	return True

# Create offsets to group consecutive words by
source = source.filter(validWord) \
			   .zipWithIndex() \
			   .map(lambda (k, v): (v, k))
sourceoffset = source.map(lambda (k, v): (k-1, v))
sourceoffset2 = source.map(lambda (k, v): (k-2, v))
# A series of transformations to create the desired RDD
final = source.join(sourceoffset) \
			  .join(sourceoffset2) \
			  .map(lambda (key, ((first, second), third)) : ((first, second, third), 1)) \
			  .grouByKey() \
			  .map(lambda (key, values) : (key, len(values))) \
			  .map(lambda ((w1, w2, w3), count) : ((w1, w2), (w3, count)))\
			  .grouByKey()\
			  .map(lambda (k, v) : (k, list(v)))

# Print out 10 sentences of 20 words
ret = []
for i in range(10):
	start = fifth.takeSample(True, 1)
	current_key = start[0][0]
	sentence = []
	for i in range(20):
		sentence.append(current_key[0])
		values = fifth.lookup(current_key)[0]
		best = max(values, key = lambda (word, count) : count)
		current_key = (current_key[1], best[0])
	ret.append(sentence)

for sentence in ret:
	for word in sentence:
		print word  + " ",
	print "\n"