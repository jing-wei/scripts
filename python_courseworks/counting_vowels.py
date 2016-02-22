# counting vowels
# suppose s is given
# vowel 'a' or 'e' or 'i' or 'o' or 'u'

cnt = 0
for letter in s:
    if letter == 'a':
        cnt = cnt +1
    if letter == 'e':
        cnt = cnt +1
    if letter == 'i':
        cnt = cnt +1
    if letter == 'o':
        cnt = cnt +1
    if letter == 'u':
        cnt = cnt +1
print cnt