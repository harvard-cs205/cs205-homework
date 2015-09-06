# -*- coding: utf-8 -*-
"""
Script for analyzing the # of seconds to count N bags when
handing out x bags per second

"""
import copy

debug = True

#Assume x divides N for simplicity
def calculateTime(N, x):
    employees = [0]*(N/x)
    seconds = 1
    for i in range(0, N/x):
        #hand out x bags to i-th employee
        employees[i] = x
        processStep(employees, i, seconds)
        seconds = seconds + 1
    while(finished(employees) != True):
        processStep(employees, N/x, seconds)
        seconds = seconds + 1
    print "Completed in " + str(seconds-1) + " seconds.\n"
        
        
def processStep(employees, i, seconds):
    #iterate through previous employees to see if they can 
    #sum numbers
    #communicate # to someone else
    #create copy so we don't confuse what happened this second to what's happened previously
    if debug:
        print "Second " + str(seconds) + "\n"
    temp = copy.copy(employees)
    for j in range(0, i):
        if temp[j] > 1:
            employees[j] = employees[j] - 1
            if debug:
                print "Employee " + str(j+1) + " sums\n"
        elif temp[j] == 1:
            index = 0
            min_index = i if i < len(employees) else len(employees)-1
            while (index < i):
                if index != j and temp[min_index] > temp[index] and temp[index] > 0:
                    min_index = index        
                index = index + 1
            if min_index != j and min_index <= i and temp[min_index] > 0:
                communicateBag(employees, j, min_index)
    if debug:
        printStep(employees)
        
def finished(employees):
    sum = 0
    for x in range(0, len(employees)):
        sum = sum + employees[x]
    return sum == 1
        
def communicateBag(employees, i, j):
    employees[i] = employees[i] - 1
    employees[j] = employees[j] + 1
    if debug:
        print "Employee " + str(i+1) + " communicates to " + str(j+1) + "\n"
    
def printStep(employees):
    header = "Empl:";
    bags = "Bags:";
    for x in range(0, len(employees)):
        if employees[x] > 0:
            header += str((x+1)) + " "
            bags += str(employees[x]) + " "
            if len(header) < len(bags):
                for y in range(0, len(bags) - len(header)):
                    header += " "
            else:
                for y in range(0, len(header) - len(bags)):
                    bags += " "
    print header + "\n" + bags + "\n\n"
            

if __name__ == '__main__':
    calculateTime(256,2)