def main():
	all_words=sc.textFile('* Words.csv')
	sorted_words=all_words.map(lambda x: (''.join(sorted(x)),x))
	anagrams=sorted_words.groupByKey().map(lambda x:(x[0],len(x[1]),list(x[1])))
	anagrams.takeOrdered(3,lambda x:-x[1])