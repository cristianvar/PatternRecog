# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 17:40:13 2022

@author: Titan
"""
import numpy

def sumBitDifferences(arr, n):
 
    ans = 0  # Initialize result
 
    # traverse over all bits. This part goes from 0 to the ith bit of the number
    #You should change this value depending on the lenght of your binary numbers
    # for example, if you want to compate 3 and 2 (binary 11 & 10 respectively)
    # i should have a value from 0 to 2. in this case is 0 to 32 to compare 32
    # numbers bit long
    for i in range(0, 32):
     
        # count number of elements with i'th bit set
        count = 0
        #this section takes the range of number you want to compare
        for j in range(0, n):
            #it performs an and (&) bitwise operation from lsb (less significant bit) to msb (most significant bit)
            #with the number 1 (01). If the corresponding column differs in both of the numbers, ans variable will add +1
            #for example comparing 01 and 11 will have 0 variation in the first column but in the second one the bit will
            #variate in 1 bit therefore the final value for ans = 1 respectively
            if ( (arr[j] & (1 << i)) ):
                count+= 1
 
        # Add "count * (n - count) * 2" to the answer
        ans += (count * (n - count) );
     
    return ans
 
# Driver program
nsymbols=36 #change it depending on the constellation size
arr = [0,0]
n = len(arr )
dicc = numpy.zeros((nsymbols,nsymbols))
 
# This code is contributed by
# Smitha Dinesh Semwal
#modified by Cristian Vargas 

#the nested for loops will fill the dicc matrix with the corresponding values of
#differing bits. This is done by comparing number by number depending on the size of the constellation 
#for example, 0 compared to 1,2,3,4.....36
#1 compared to 0,2,3,4.....36 and so on
for x in range(0,nsymbols):
    for y in range(0,nsymbols):
        arr =[x,y]
        dicc[x,y]=sumBitDifferences(arr, n)


#the final matrix is stored in a CSV file to be utilized later.
numpy.savetxt('36BER_DIC.csv', dicc, delimiter=',')