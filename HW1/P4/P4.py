# P4 solution
import csv
sourcefile = 'source.csv'

from pyspark import SparkConf, SparkContext
conf = SparkConf().setAppName('KaiSquare')
sc = SparkContext(conf = conf)

def findtouch(charname, charadjlist):
    result = sc.parallelize([(charname, 0)])
    accum = sc.accumulator(0)
    while True:
        curvalue = accum.value
        waiting = result.filter(lambda kv: kv[1] == curvalue)
        if waiting.count() == 0:
            break
        coming = waiting.join(charadjlist)
        coming = coming.flatMap(lambda kv: [(v, curvalue+1) for v in kv[1][1]])
        result = result.union(coming.distinct())
        result = result.reduceByKey(lambda a,b: min(a,b))
        accum.add(1)
    return result

if __name__  == '__main__':
    comiclist = []
    with open(sourcefile, 'rb') as csvfile:
        comicdata = csv.reader(csvfile, delimiter=',', quotechar='"')
        for character, issue in comicdata:
            comiclist.append((issue, character))
    comiclist = sc.parallelize(comiclist)
    issuelist = comiclist.groupByKey().mapValues(list)

    charlist = issuelist.map(lambda i: i[1])
    geteach = lambda seq: [(seq[i], seq[:i] + seq[i+1:]) for i in range(len(seq))]
    charcharlist = charlist.flatMap(geteach)

    charadjlist = charcharlist.reduceByKey(lambda a,b: a+b)
    charadjlist = charadjlist.map(lambda c: (c[0], list(set(c[1]))))
    
    for charname in ["CAPTAIN AMERICA", "MISS THING/MARY", "ORWELL"]:
        reachlist = findtouch(charname, charadjlist)
        print charname, reachlist.count()
