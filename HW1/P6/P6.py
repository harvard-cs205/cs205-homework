import pyspark

def quiet_logs(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("amazonaws").setLevel(logger.Level.WARN)

sc = pyspark.SparkContext()
quiet_logs(sc)

data = sc.textFile('Shakespeare.txt')
data = data.flatMap(lambda l: l.split())

# Data cleaning and indexing.
def clean(l):
    def isCap(l):
        for i in l:
            if (not (i.isupper())) and i!='.':
                return False
        return True
    if l.isdigit() or isCap(l):
        return False
    return True
data = data.filter(clean)	
data = data.zipWithIndex()	

# Group all the phrases made of three consecutive words
# para all_data is a list of tuples
part1 = data.map(lambda l: ((int)(l[1]/3), l[0]))
part1 = part1.groupByKey().map(lambda l: list(l[1]))
part2 = data.map(lambda l: ((int)((l[1]+1)/3), l[0]))
part2 = part2.groupByKey().map(lambda l: list(l[1]))
part3 = data.map(lambda l: ((int)((l[1]+2)/3), l[0]))
part3 = part3.groupByKey().map(lambda l: list(l[1]))

all_data = part1.union(part2)
all_data = all_data.union(part3)
all_data = all_data.filter(lambda l: len(l)==3)

# count all the following word and 
word_connections = all_data.map(lambda l: ((l[0],l[1]),l[2]))
word_connections = word_connections.groupByKey().map(lambda l: (l[0], list(l[1])))
word_connections = word_connections.map(lambda l: (l[0], [(i, l[1].count(i)) for i in set(l[1])]))
word_connections.persist()
# Use filter and I found "Now is" matches exactly in the homework.


start_words = word_connections.takeSample(False, 10)
start_words = [i[0] for i in start_words]
sentence = dict()

for key in start_words:
    sentence[key] = key[0] +' '+ key[1]
    newkey = key
    for size in range(18):# generate a 20-word size sentence
        try:
            candidates = word_connections.filter(lambda l: l[0]==newkey).collect()# here collect only one element
            candidates = dict(candidates[0][1])
            inverse = [(v, k) for k, v in candidates.items()]
            best_candidate =  max(inverse)[1]#get the highest frequency, if the numbers are the same, just use the last one.
            sentence[key] = sentence[key]+ ' ' + str(best_candidate)
            newkey = (newkey[1], best_candidate)
        except:
            break

print sentence