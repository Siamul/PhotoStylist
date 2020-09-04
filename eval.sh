#!/bin/sh
a=1

while [ $a -lt 52 ]
do
    evalFolder="./eval/"
    contentFile="content$a.jpg"
    s="style"
    styleFileList=$(find "$evalFolder" -name "content$a$s*.txt")
    for i in $styleFileList; do
        python3 poetic_style_transfer.py --content_image "$evalFolder$contentFile" --style_text "$i"
    done
    a=`expr $a + 1`
done
