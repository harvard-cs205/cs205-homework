# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 13:46:04 2015

@author: Balthasar
"""

# 1. Load list of all words into RDD
# 2. Create KEY for every word, where KEY is given by the letters of the word in alphabetical order. Anagrams have same KEY.
# 3. Reduce by key, merging the list of words together.

# import numpy
import numpy as np

# load wordlist
wlist = sc.textFile('s3://Harvard-CS205/wordlist/EOWL_words.txt')

# create rdd with the key being the words having their letters rearranged alphabetically (i.e. bear -> aber)
wlistKey = wlist.map(lambda x: (" ".join(x),x)) # insert spaces between letters: 'bear' -> 'b e a r' 
wlistKey = wlistKey.map(lambda x: (x[0].split(" "),x[1])) # split where there's spaces: 'b e a r' -> 'b' 'e' 'a' 'r'
wlistKey = wlistKey.map(lambda x: (np.sort(x[0]),x[1])) # 'b' 'e' 'a' 'r' -> 'a' 'b' 'e' 'r'

#TESTING:
#>>> wlist.take(5)
#
#[u'aa', u'aah', u'aal', u'aalii', u'aardvark']
#
#>>> wlistKey.key().take(5)
#
#[array([u'a', u'a'], 
#      dtype='<U1'), array([u'a', u'a', u'h'], 
#      dtype='<U1'), array([u'a', u'a', u'l'], 
#      dtype='<U1'), array([u'a', u'a', u'i', u'i', u'l'], 
#      dtype='<U1'), array([u'a', u'a', u'a', u'd', u'k', u'r', u'r', u'v'], 
#      dtype='<U1')]
#     

# need to get rid of the array numpy datatype because rdd reduce can't deal with it

wlistKey = wlistKey.map(lambda x: (x[0].tolist(),x[1])) # reduce doesn't like the numpy array datatype
wlistKey = wlistKey.map(lambda x: ("".join(x[0]),x[1]))
anagrams = wlistKey.groupByKey() # returns an iterable
anagrams = anagrams.map(lambda x: (x[0],list(x[1]))) # returns a list
anagrams = anagrams.map(lambda x: (len(x[1]),x[0],x[1])) # add number of anagrams as key

# >>> anagrams.countByKey()  # number of anagrams of each kind
# defaultdict(<type 'int'>, {1: 100008, 2: 9441, 3: 1998, 4: 581, 5: 179, 6: 78, 7: 34, 8: 12, 9: 3, 10: 2, 11: 3})

# >>> anagrams.take(50)
#[(1, u'aacikrtu', [u'autarkic']), (1, u'eelossttx', [u'sextolets']), (1, u'eiilrsttw', [u'twirliest']), (1, u'deenoprrsv', [u'provenders']), (1, u'ceilnrstu', [u'linctures']), (1, u'aehhrst', [u'hearths']), (1, u'iilps', [u'pilis']), (1, u'dgiinoww', [u'widowing']), (1, u'aacddelop', [u'decapodal']), (1, u'ehostw', [u'theows']), (1, u'degillnw', [u'dwelling']), (1, u'aceiimrsst', [u'armistices']), (1, u'degillnu', [u'duelling']), (1, u'eiills', [u'lilies']), (1, u'ddgiiinsv', [u'dividings']), (1, u'cenrssuw', [u'unscrews']), (1, u'aafilstt', [u'fatalist']), (1, u'abbelmr', [u'bramble']), (1, u'deimrsstty', [u'mistrysted']), (1, u'acefhipry', [u'preachify']), (1, u'abehrrs', [u'brasher']), (1, u'deimrssttu', [u'mistrusted']), (1, u'eeiinrstwz', [u'winterizes']), (1, u'abehrry', [u'herbary']), (1, u'eeilnosvz', [u'novelizes']), (1, u'aacent', [u'catena']), (1, u'abcehlrsu', [u'crushable']), (1, u'adeilmmnt', [u'immantled']), (1, u'dginou', [u'guidon']), (1, u'cdehipt', [u'pitched']), (1, u'cceiklr', [u'clicker']), (1, u'emrst', [u'terms']), (1, u'aabdlnpst', [u'platbands']), (1, u'dginsu', [u'dingus']), (3, u'aeknrs', [u'nakers', u'nerkas', u'ankers']), (1, u'aeknru', [u'unrake']), (1, u'aeknrw', [u'wanker']), (1, u'eiilrstty', [u'sterility']), (1, u'bbbelo', [u'bobble']), (1, u'adeellppr', [u'rappelled']), (2, u'deorrst', [u'rodster', u'dorters']), (1, u'anosxy', [u'saxony']), (1, u'cefflrssu', [u'scufflers']), (1, u'ceghiiost', [u'gothicise']), (1, u'egmnoru', [u'murgeon']), (4, u'ehors', [u'shoer', u'shore', u'hoers', u'horse']), (1, u'cghioprsty', [u'copyrights']), (1, u'dgimmnuy', [u'dummying']), (1, u'cceeeennss', [u'senescence']), (1, u'bgiijnnosu', [u'subjoining']), (1, u'egmnory', [u'mongery']), (1, u'ciimnot', [u'miction']), (1, u'denoprss', [u'responds']), (1, u'aeilprxy', [u'pyrexial']), (1, u'aimossstt', [u'somatists']), (1, u'cgiilmnuu', [u'glucinium']), (1, u'eefllostuw', [u'woefullest']), (1, u'aacglost', [u'catalogs']), (1, u'aaddegiilt', [u'digladiate']), (1, u'egmnnooosu', [u'monogenous'])]

#anagramsEleven = anagrams.filter(lambda x: x[0] == 11 ).collect()

#>>> anagramsEleven
#[(11, u'aerst', [u'arets', u'aster', u'rates', u'reast', u'resat', u'stare', u'stear', u'strae', u'tares', u'tears', u'teras']), (11, u'aelst', [u'least', u'leats', u'salet', u'slate', u'stale', u'steal', u'stela', u'taels', u'tales', u'teals', u'tesla']), (11, u'aeprs', [u'asper', u'pares', u'parse', u'pears', u'prase', u'presa', u'rapes', u'reaps', u'spaer', u'spare', u'spear'])] 

# The letter combinations 'aerst','aelst' and 'aeprs' have 11 anagrams. 