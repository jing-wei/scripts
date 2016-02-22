'''
Write a Python function that returns the sum of the pairwise products of 
listA and listB. You should assume that listA and listB have the same length 
and are two lists of integer numbers. For example, if listA = [1, 2, 3] and 
listB = [4, 5, 6], the dot product is 1*4 + 2*5 + 3*6, meaning your function 
should return: 32

Hint: You will need to traverse both lists in parallel.

This function takes in two lists of numbers and returns a number.
'''
def dotProduct(listA, listB):
    list_length = len(listA)
    list_product = 0
    for i in range(list_length):
        list_product += listA[i] * listB[i]
    return list_product