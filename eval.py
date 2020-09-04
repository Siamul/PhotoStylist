import os
import sys

for i in range(51):
    j = i+1
    content_file = './eval/content'+str(j)+'.jpg'
    style_file = './eval/style'+str(j)+'.txt'
    with open(style_file, 'r') as sfile:
        k=1
        for line in sfile:
            line = line.strip()
            with open('./eval/content'+str(j)+'style'+str(k)+'.txt', 'w') as sfile:
                sfile.write(line)
            k+=1
