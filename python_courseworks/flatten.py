'''
Write a function to flatten a list. The list contains other lists, 
strings, or ints. For example, [[1,'a',['cat'],2],[[[3]],'dog'],4,5] 
is flattened into [1,'a','cat',2,3,'dog',4,5]
'''



def flatten(aList):
    ''' 
    aList: a list 
    Returns a copy of aList, which is a flattened version of aList 
    '''
    tmp = aList
    output = list()
    for i in range(len(tmp)):
        if type(tmp[i]) == str or type(tmp[i]) == int:
            output.append(tmp[i])
        else:
            for j in range(len(tmp[i])):
                if type(tmp[i][j]) == str or type(tmp[i][j]) == int:
                    output.append(tmp[i][j])
                else:
                    flatten(tmp[j])
    return output
    