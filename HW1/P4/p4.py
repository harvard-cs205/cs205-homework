def main():
	#CSV takes the file and creates and RDD of character,issue
	csv=(sc.textFile('source.csv').map(lambda x: x.split('","'))
	                                .map(lambda x:[x[0][1:],x[1][:-1]]))
	csv.cache()
	#issues_by_character groups the issues with character as key
	issues_by_character=csv.groupByKey().map(lambda x:[x[0],list(x[1])])
	#characters_by_issue creates a dict with issue as key and list of chars as value
	characters_by_issue=csv.map(lambda x:[x[1],x[0]]).groupByKey().map(lambda x:[x[0],list(x[1])]).collectAsMap()
	#linked characters is an RDD that has the x[0] as the char name and a list of chars its linked to as x[1]
	linked_chars=(issues_by_character
	              .map(lambda x:[x[0],[characters_by_issue[issue] for issue in x[1]]])
	              .map(lambda x:[x[0],[char for chars in x[1] for char in chars]])
	              .map(lambda x:[x[0],list(set(x[1]))])
	              .map(lambda x:[x[0],[a for a in x[1] if a != x[0]]]))
	linked_chars.cache()
	node_depths=BFS(linked_chars,'CAPTAIN AMERICA')
	node_depths.count() #6408
