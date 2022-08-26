# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 17:40:13 2022

@author: Titan
"""
import numpy

def sumBitDifferences(arr, n):
 
    ans = 0  # Initialize result
 
    # traverse over all bits
    for i in range(0, 32):
     
        # count number of elements with i'th bit set
        count = 0
        for j in range(0, n):
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
print(sumBitDifferences(arr, n))
 
# This code is contributed by
# Smitha Dinesh Semwal   

for x in range(0,nsymbols):
    for y in range(0,nsymbols):
        arr =[x,y]
        dicc[x,y]=sumBitDifferences(arr, n)

numpy.savetxt('36BER_DIC.csv', dicc, delimiter=',')