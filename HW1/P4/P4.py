from pyspark import SparkContext
from P4_bfs import bfs

if __name__ == "__main__":
    
    # initialize SparkContext
    sc = SparkContext(appName="Marvel Character Map")
    
    # read text file into lines
    lines = sc.textFile("marvel.txt")
    
    # strip beginning and ending ", then split the csv into a two-element list based on middle ","
    charList = lines.map(lambda line: line.strip("\"").split("\",\""))
    
    # rearrange charList as tuples with issue as key and character as value, then group by issue (to get a set of characters appearing in each issue)
    issueToChars = charList.map(lambda list: (list[1], list[0])).groupByKey().mapValues(set)
    
    # rearrange charList as tuples with issue as key and character as value
    issueToChar = charList.map(lambda list: (list[1], list[0]))
    
    # join (issue, character) and (issue, character set) to get (issue, (character, character set)), then isolate the value tuple
    charMap = issueToChar.join(issueToChars).map(lambda tup: tup[1])
    
    # reduce the character map to contain a unique set of values
    charMap = charMap.reduceByKey(lambda set1, set2: set1 | set2).sortByKey()
    
    bfs(charMap, "CAPTAIN AMERICA")
    
    sc.stop()
