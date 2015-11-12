import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import seaborn as sns
sns.set_context('poster', font_scale=1.25)
from random import randint

# initializing spark
import pyspark as ps

config = ps.SparkConf()
config = config.setMaster('local[' + str(2*mp.cpu_count()) + ']')
config = config.setAppName('P6')

sc = ps.SparkContext(conf=config)
# REMOVE ALL OF THE DIFFERENT SPARK WARNINGS
sc.setLogLevel('WARN')

# IMPORT THE TEXT FILE CONTAINING THE PLAY. 
dataSet=sc.textFile('Shakespeare.txt')

# CLEAN THE DATA AND REMOVE UNWANTED CHARACTER TYPES AS WELL AS DIGITTS AND SPLIT EACH 
# SENTENCE INTO DISCRETE WORDS. THIS CLEANS UP THE WORDS FROM THE TEXTFILE
def cleanData(x):
    x=x.split()
    words=[]
    for i in xrange(0,len(x)):
        if (len(x[i]) is not 0) and (not x[i].isupper()) and (not x[i].isdigit()) and (x[i]!= ' '):
            words.append(x[i])
    return words

# EXECUTE THE FUNCTION TO CLEAND THE DATA AS DEFINED ABOVE
cleanDataSet=dataSet.flatMap(lambda x: cleanData(x)).zipWithIndex().map(lambda x: (x[1],x[0]))

# CREATE A SHIFTED DATASET WITH ALL WORDS SHIFTED 1 POSITION
# AND TWO POSTIIONS RESPECTIVELY
shift_1=cleanDataSet.map(lambda x: (x[0]-1,x[1]))
shift_2=cleanDataSet.map(lambda x: (x[0]-2,x[1]))

# COMBINE THE DATASETS ABOVE USING THE JOIN FUNCTION. THEN CLEAND AN CREATE
combined_rdd = cleanDataSet.join(shift_1).join(shift_2).map(lambda x:((x[1][0][0], x[1][0][1], x[1][1]),1))
combined_rdd=combined_rdd.reduceByKey(lambda x,y: x+y)
groups_of_words = combined_rdd.map(lambda x: ((x[0][0],x[0][1]),(x[0][2],x[1])))
# PRINT THE OUTPUT FOR 'NOW IS' AS SUGGESTED BY THE HOMEWORK IN ORDER TO ENSURE
# THAT WE ARE GETTING THE SAME THING AS THE PROBLEM SET
print groups_of_words.lookup(('Now', 'is'))

# CREATE THE BIAS FOR FINDING THE MOST POPULAR WORD; IF THEY ARE OF EQUAL 
# POPULARITY USE A RANDOM CHOICE FOR THE NEXT WORD, THIS WILL AVOID THE 
# PHRASE TO KEEP GOING IN LOOPS AND GETTING STUCK ON THE SAME FEW WORDS
def findPopularWord(x):
    highest=0
    theword=''
    for word in x:
        if word[1]>=highest:
            highest=word[1]
            theword=word
    if highest==1:
        theword=x[randint(0,len([1 for _ in x]))-1]
    return theword

# FUNCTION TO GENEARATE THE PHRASES, BY PASSING RDD WITH THE WORDS INSIDE
# THIS WILL RUN THE POPULAR WORDS FUNCTION ABOVE AND WILL APPEND THE NEXT 
# WORD ONTO THE SENTENCE. THIS GOES ON A LOOP UNTIL THE DESIRED SENTENCE 
# SIZE IS OBTAINED.
def generatePhrase(RDD,numWords=20):
    RDD=RDD.map(lambda x: x)
    sample=RDD.takeSample(False,1)[0]
    word=RDD.lookup((sample[0][0],sample[0][1]))
    nextWord=findPopularWord(word)[0]
    phrase=[sample[0][0],sample[0][1],nextWord]
    phraseLength=3
    print "Initial Sample Phrase:",phrase
    while phraseLength<numWords:
        phraseLength+=1
        print "Current Phrase Length: ",phraseLength, " of ",numWords
        word=RDD.lookup((phrase[-2],phrase[-1]))
        #print word
        phrase.append(findPopularWord(word)[0])
    return phrase

# DEFINE THE NUMBER OF PHRSES DESIRED
numberOfPhrases=10
# CREATE EMPTY PHRASE LIST TO BE PUPULATED WITH THE WORDS FOR THE DIFFERENT
# PHRASES. 
phrases=[]

# ITTERATE OVER THE PHRASES TO OBTAIN THE NUMBER OF PHRASES DESIRED WITH THE 
# NUMBER OF WORDS DESIRED
for phraseNum in range(0,numberOfPhrases):
    print "Phrase Number:",phraseNum+1, " of ",numberOfPhrases
    currentPhrase=generatePhrase(groups_of_words)
    phrases.append(currentPhrase)

# PRINT EACH WORD IN A PHRASE WITH A SPACE AND PRINT EACH PHRASE SEPARATELY
for words in phrases:
    for word in words:
        print word,
    print "\n"
    