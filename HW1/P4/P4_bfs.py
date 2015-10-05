def SSBFS(RDD,character_name):
	# CREATE AN RDD THAT HAS ALL OF THE CHARACTERS AND THEIR CONNECTIONS
	# GET INITIAL COUNT OF THE RDD SO THAT WE CAN COMPARE HOW MANY HAVE CHANGED
	charsList=RDD
	charsListNumFirst=charsList.count()
	charsListNumOld=charsListNumFirst

	# FIND THE CHARACTER DESIRED FROM THE FULL CHARACTER RDD, THIS WILL ALSO
	# GIVE US ALL OF THEIR RESPECTIVE CONNECTIONS AS WELL
	char=charsList.filter(lambda (k,v): k==character_name)

	# OBTAIN A LIST OF ARRAYS THAT CONTAINS THE CHARACTERS TO BE FILTERED 
	tempList=char.values().collect()[0]
	# COMPARE THE LIST USING A FILTER TO OBTAIN THE LIST OF NEW NODES WHICH IS THE
	# NEXT LEVEL FOR THE SEARCH
	NextNodes=charsList.filter(lambda (k,v): k in tempList).flatMapValues(lambda x: x)
	NextNodes=NextNodes.map(lambda(k,v): (v,1)).groupByKey().map(lambda(k,v): (k,1))
	
	# SUBTRACT THE SEARCHED LIST FROM THE TOTAL LIST OF CHARACTERS THAT KEEPS
	# TRACK OF WHICH CHARACTERS HAVE BEEN SEARCHED
	charsList=charsList.subtractByKey(char)
	charsListNumNew=charsList.count()

	# SET THE ITERATION TO 0 FOR THE SEARCH AND PRINT CURRENT STATE
	
	iteration=sc.accumulator(0)
	print "##BEGIN##"
	print "Iteration:",iteration
	print "New Characters:",charsListNumNew
	print "Old Characters:",charsListNumOld

	# ITTERATE UNTIL THE NUMBER IN THE LIST OF CHARACTERS STOPS CHANGING, MEANING
	# THAT THE OLD LIST IS NOW EQUAL TO THE NUMBER OF NEW LIST THIS IS WHY THE ITRATION 
	# IS INITIALY SET TO 0, AS IT WILL TAKE ONE MORE STEP AND TAHT STEP WILL BE ACCOUNTED FOR 
	# IN THE ITERATION VARIABLE
	while charsListNumOld>charsListNumNew:
		# RETAIN THE OLD NODES SO THAT WE CAN SUBTRACTBYKEY() LATER ON
	    oldNodes=NextNodes
	    # DO THE OPERATION ONCE AGAIN AS DONE OUTSIDE OF THE LOOP
	    # THE REASON THAT OTHER OPERATIONS ARE KEPT OUTSIDE OF THE LOOP IS THAT
	    # I NEEDED TO OBTAIN THE FIRST CHARACTER AND DO THE OPERATION ON THAT 
	    # BEFORE I COULD PERFOMR AN ITERATIVE ANALYSIS
	    tempList=NextNodes.keys().collect()
	    NextNodes=charsList.filter(lambda (k,v): k in tempList).flatMapValues(lambda x: x)
	    NextNodes=NextNodes.map(lambda(k,v): (v,1)).groupByKey().map(lambda(k,v): (k,1))

	    # SUBTRACT THE OLD NODES FROM THE TOTAL LIST OF CHARACTERS
	    charsList=charsList.subtractByKey(oldNodes)
	    # UPDATE THE CHARACTER NUMBERS WITH THE NEW LIST NUMBERS AND UPDATE
	    # THE OLD NUMBER WITH THE PREVIOUS NUMBER OF CHARACTERS ON THE LIST
	    charsListNumOld,charsListNumNew=charsListNumNew,charsList.count()

	    # PRINT CURRENT STATE FOR DEBUGGING AND PROGRESS REASONST
	    iteration+=1
	    print "########"
	    print "Iteration:",iteration
	    print "New Characters:",charsListNumNew
	    print "Old Characters:",charsListNumOld
	iteration=iteration.value-1
	# PRINT THE FINAL STATE OF THE ANALYSIS WITH THE OUTPUT REQUESTED
	print"###DONE###"
	print "Iteration:",iteration
	print "Number of Characters Not Touched:", charsListNumFirst-charsListNumOld

	return iteration,charsListNumNew,charsListNumFirst