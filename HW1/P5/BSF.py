# import findspark
# findspark.init()
import pyspark
sc = pyspark.SparkContext('local[16]')
N = 32


def link_string_to_KV(s):
	    src, dests = s.split(': ')
	    dests = [int(to) for to in dests.split(' ')]
	    return (int(src), dests)

def filterer(i):
	    def filt(x):
	        return x==itr_acc.value - 1
	    return filt

def get_first_non_empty_cogroup(x2):
    for elem in x2:
        elem = list(elem)
        if len(elem) > 0:
            return elem[0]

def filterer(i):
    def filt(x):
        return x==itr_acc.value - 1
    return filt

def get_first_non_empty_cogroup(x2):
    for elem in x2:
        elem = list(elem)
        if len(elem) > 0:
            return elem[0]

def shortest_path(graph, nodeA, nodeB):
    

    nodes = sc.parallelize([(nodeA, 0)])
    node_destination = [nodeB]
    # a = a.map(lambda x: (x[0], (list(x[1][0]))[0], (list(x[1][1]))))

    total_char_acc = sc.accumulator(1)
    it_acc = sc.accumulator(0)
    nodes_4 = sc.parallelize([])


    while True:
        it_acc += 1
        dist = it_acc.value
        #     nodes = a.filter(lambda x: x[2] == [dist]).flatMap(lambda x: [(i, dist + 1) for i in x[1]]).distinct()

        nodes_1 = nodes.filter(lambda x: filterer(x[1])).join(graph).partitionBy(N)

        nodes_2 = nodes_1.flatMap(lambda x: x[1][1]).distinct().map(lambda x: (x, dist))

        nodes_3 = nodes_1.partitionBy(N).map(lambda x: (x[0],x[1][1],dist))

        nodes_3 = nodes_3.map(lambda x: [(x[0], i, dist) for i in x[1]])

        nodes_4 = nodes_3.union(nodes_4)


        nodes = nodes.cogroup(nodes_2).map(lambda x: (x[0], get_first_non_empty_cogroup(list(x[1])))).cache()

        
        if nodes_2.map(lambda x: x[0]).filter(lambda x: node_destination[0] == x).count() > 0:
            print "Distance:", dist
            break


    path = [node_destination]
    nodes_4 = nodes_4.flatMap(lambda x: x)

    for i in range(dist,0,-1):

        node_destination = nodes_4.filter(lambda x: x[2] == i and x[1] in node_destination[:]).map(lambda x: x[0]).distinct().collect()
    
        path.append(node_destination)

    return dist, path



def transformToWords(b):
    path = []
    for i in b:
        if len(i)>1:

            path.append([page_names.lookup(j)[0] for j in i])
        else:

            path.append(page_names.lookup(i[0])[0])
    return path

if __name__ == '__main__':


	links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
   	page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')

	# links = sc.textFile('./testdata.txt')
	# page_names = sc.textFile('./pages.txt')
	# process links into (node #, [neighbor node #, neighbor node #, ...]

	neighbor_graph = links.map(link_string_to_KV).cache()

	page_numbers = page_names.zipWithIndex().map(lambda x:(x[0],x[1]+1))
	Kevin = page_numbers.lookup('Kevin Bacon')[0]
	Harvard = page_numbers.lookup('Harvard University')[0]

	page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
	page_names = page_names.sortByKey().cache()

	
	[distKH, pathKH] = shortest_path(neighbor_graph, Kevin, Harvard)

	[distHK, pathHK] = shortest_path(neighbor_graph,  Harvard, Kevin)

	print "The distance from Harvard University to Kevin Bacon is", distHK,"and the possible paths are:", transformToWords(pathHK),". The distance from Kevin Bacon to Harvard University is", distKH,"and the possible paths are:", transformToWords(pathKH)





#The distance from Harvard University to Kevin Bacon is 3 and the possible paths are:[u'Kevin_Bacon', [u'Edmund_Bacon', u'Kyra_Sedgwick', u'John_Edwards_presidential_campaign,_2008', u'Mister_Roberts_(film)', u'Daniel_Bigel', u'Litchfield_Hills', u'Footloose', u'2007_in_film', u'Apollo_13', u"Paris,_je_t'aime", u'Screen_Actors_Guild_Award_for_Outstanding_Performance_by_a_Cast_in_a_Motion_Picture', u'Gene_Hackman', u'Tom_Hanks', u'Endicott_Peabody_(educator)', u"National_Lampoon's_Animal_House", u'Elvis_Costello', u'Dennis_Lehane', u'Mickey_Rourke', u'Metra', u'The_River_Wild', u'2005_Cannes_Film_Festival', u'Rod_Steiger', u'Liza_Weil', u'1958', u'2003_in_film', u'Jack_Swigert', u'Dustin_Hoffman', u'Judith_Miller_(journalist)', u'Mad_(magazine)', u'Brad_Renfro', u'New_York,_I_Love_You', u'Erd\u0151s\u2013Bacon_number', u'Tiki_Barber', u'Live_Earth_(2007_concert)', u'WFUV', u'South_Boston,_Boston,_Massachusetts', u'2008_in_film', u'JFK_(film)', u'Massachusetts_State_Police', u'2006_in_film', u'Thriller_(genre)', u'Christina_Applegate', u'A_Few_Good_Men', u'July_8', u'Empire_State_Building'], [u'Audioslave', u'May_16', u'Tommy_Lee_Jones', u'September_2', u'The_New_York_Times', u'2008', u"Conan_O'Brien", u'The_Great_Debaters', u'New_England_Patriots', u'Gilmore_Girls', u'The_Da_Vinci_Code', u'John_Lithgow', u'Al_Gore', u'Jamaica_Plain,_Massachusetts', u'September_28', u'Charles_River', u'Charlestown,_Massachusetts', u'The_Firm_(1993_film)', u'February_19', u'Cornell_University', u'New_England', u'Matt_Damon', u'Syracuse_University', u'Hurricane_Katrina', u'2007', u'September_15', u'Good_Will_Hunting', u'Washington,_D.C.', u'Boston,_Massachusetts', u'Dartmouth_College', u'September_16', u'London', u'List_of_U.S._colleges_and_universities_by_endowment', u'Longwood_Medical_and_Academic_Area', u'Quincy_College', u'Pine_Manor_College', u'Thomas_Dudley', u'John_Kerry', u'Hasty_Pudding_Man_of_the_Year', u'Harvard_Law_School', u'Hasty_Pudding_Theatricals', u'Hasty_Pudding_Woman_of_the_Year', u'Rutgers_University', u'Plymouth,_Massachusetts', u'Harvard_Man', u'Robert_Benchley', u'Columbia_University', u'The_Yale_Record', u'February_26', u'Northwestern_University', u'March_20', u'Richard_Nixon', u'Massachusetts_Bay_Colony', u'University_of_Virginia', u'2003', u'2006', u'Tram', u'National_Lampoon_(magazine)', u'University_of_Hartford', u'Boston_Brahmin', u'George_W._Bush', u'Texas_A&M_University', u'University_of_Oregon', u'Tulane_University', u'Legally_Blonde', u'Fordham_University', u'August_30', u'September_8', u'2004', u'1998', u'May_21', u'N._Gregory_Mankiw', u'Jack_Lemmon', u'Georgetown_University', u'Blasphemy_(novel)', u'Natalie_Portman', u'Andy_Borowitz', u'Franklin_D._Roosevelt', u'Tyrannosaur_Canyon', u'University_at_Buffalo,_The_State_University_of_New_York', u'Dan_Brown'], u'Harvard_University'] . 

#The distance from Kevin Bacon to Harvard University is 2 and the possible paths are: [u'Harvard_University', [u'Time_(magazine)', u'College_Bowl', u'John_Lithgow', u'Marisa_Silver', u'Six_degrees_of_separation'], u'Kevin_Bacon']

	
