from pyspark import SparkContext
from urllib2 import urlopen
import os

if __name__=='__main__':
    # initialize spark
    sc = SparkContext()
    sc.setLogLevel('ERROR')

    # save text file
    datafile = 'source.csv'
    if not os.path.exists(datafile):
        # download text file
        url = 'http://exposedata.com/marvel/data/source.csv'
        with open(datafile, 'wb') as f:
            f.write(urlopen(url).read())

    # load data into a rdd (issue character)
    data = sc.textFile(datafile).map(
        lambda line: tuple([w.strip('"').strip() for w in 
            line.split('","')])[::-1])



    # print data.takeSample(False, 20)
