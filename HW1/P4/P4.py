import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import seaborn as sns
execfile('P4_bfs.py')
sns.set_context('poster', font_scale=1.25)
from random import randint

# initializing spark
import pyspark as ps

config = ps.SparkConf()
config = config.setMaster('local[' + str(2*mp.cpu_count()) + ']')
config = config.setAppName('P4')

sc = ps.SparkContext(conf=config)

# importing the data set from the .csv file and removing the superfulous commas and quotations
# furthermore, this orders the dataSet RDD such that commic issue is the key and the character name is the value
dataSet=sc.textFile('source.csv').map(lambda line: line.split('","')).map(lambda x: (x[1].replace('"',''),x[0].replace('"','')))

# multply around the people so that every combination for each book exists
dataSet=dataSet.join(dataSet)

# filter for references that have two repeated characters and then map all of the connections over and group them by the
# first charachter, which through the previous opration we should have all of the different combination of characters
# furthermore, organize all of the charachters by alphabetical order and make the character's conection into list rather than iterable object
connections=dataSet.filter(lambda x: x[1][0]!=x[1][1]).map(lambda x: x[1]).groupByKey().sortByKey().mapValues(list)

####### PRINT CHARACTERS FOR DEBUGGING AND ANALYSIS ######

numCharacters=1 # number of outputs for the connection of number of characters
# print function for the connections of each character
for x,y in connections.take(numCharacters):
    print 'Character:',x
    print 'Number of Connections:',len(y)
    print 'Connections:',
    for yy in y:
        print yy,';',
    print '\n'

###############################


############ DO THE ANALYSIS FOR CAPTIAN AMERICA#########
characterName="CAPTAIN AMERICA"
iteration,charsListNumNew,charsListNumFirst=SSBFS(connections,characterName)

print "Character's Name: ", characterName
print "Iteration: ",iteration
print "New Number of Characters: ",charsListNumNew
print "First Character Numbers: ", charsListNumFirst
print "Number of Characters Not Touched: ", charsListNumFirst-charsListNumOld


############ DO THE ANALYSIS FOR MISS THING/MARRY#########
characterName="MISS THING/MARY"
iteration,charsListNumNew,charsListNumFirst=SSBFS(connections,characterName)

print "Character's Name: ", characterName
print "Iteration: ",iteration
print "New Number of Characters: ",charsListNumNew
print "First Character Numbers: ", charsListNumFirst
print "Number of Characters Not Touched: ", charsListNumFirst-charsListNumOld


############ DO THE ANALYSIS FOR ORWELL#########
characterName="ORWELL"
iteration,charsListNumNew,charsListNumFirst=SSBFS(connections,characterName)

print "Character's Name: ", characterName
print "Iteration: ",iteration
print "New Number of Characters: ",charsListNumNew
print "First Character Numbers: ", charsListNumFirst
print "Number of Characters Not Touched: ", charsListNumFirst-charsListNumOld

