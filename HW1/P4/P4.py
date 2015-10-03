# P4 solution
import csv
sourcefile = 'source.csv'

from P4_bfs import ssbfs

from pyspark import SparkConf, SparkContext
conf = SparkConf().setAppName('KaiSquare')
sc = SparkContext(conf = conf)

if __name__  == '__main__':
    comiclist = sc.textFile(sourcefile)  # take in the csv
    comiclist = comiclist.map(lambda line: tuple(line[1:-1].split('","')[::-1]))  # manipulate with the line
    issuelist = comiclist.groupByKey().mapValues(list)  # group by issue, put the values in a list

    charlist = issuelist.map(lambda i: i[1])  # for each issue, get the character list
    geteach = lambda seq: [(seq[i], seq[:i] + seq[i+1:]) for i in range(len(seq))]  # map from a character list to one character out of it to the rest of characters
    charcharlist = charlist.flatMap(geteach) # it is now adjacent list

    charadjlist = charcharlist.reduceByKey(lambda a,b: a+b) # combine adjacent list
    charadjlist = charadjlist.map(lambda c: (c[0], list(set(c[1]))))  # get unique values
    
    for charname in ["CAPTAIN AMERICA", "MISS THING/MARY", "ORWELL"]:
        reachlist = ssbfs(charname, charadjlist)
        print charname, reachlist.count()
