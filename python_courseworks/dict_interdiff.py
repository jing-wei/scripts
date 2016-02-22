def dict_interdiff(d1, d2):
    '''
    d1, d2: dicts whose keys and values are integers
    Returns a tuple of dictionaries according to the instructions above
    '''
    inter = {}
    diff = {}
    for i in range(len(d1.keys())):
        if d1.keys()[i] in d2.keys():
            inter[d1.keys()[i]] = f(d1[d1.keys()[i]], d2[d1.keys()[i]])
        else:
            diff[d1.keys()[i]] = d1[d1.keys()[i]]
    for j in range(len(d2.keys())):
        if not(d2.keys()[j] in inter.keys()):
            diff[d2.keys()[j]] = d2[d2.keys()[j]]
    return (inter, diff)