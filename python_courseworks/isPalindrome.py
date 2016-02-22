'''
Write a Python function that returns True if aString is a palindrome 
(reads the same forwards or reversed) and False otherwise. 
Do not use Python's built-in reverse function or aString[::-1] to reverse strings.

This function takes in a string and returns a boolean.
'''
def isPalindrome(aString):
    aStringReversed = ''
    for i in range(len(aString)-1,-1,-1):
        aStringReversed += aString[i]
    if aString == aStringReversed:
        return True
    else:
        return False
