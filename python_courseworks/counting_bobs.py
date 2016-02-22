# counting bobs
cnt = 0
for i in range(0, len(s)):
    if s[i:(i+3)] == 'bob':
        cnt = cnt + 1
print cnt