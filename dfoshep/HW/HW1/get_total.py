import sys
import string



verbal = 0
math = 0
count=0
for line in sys.stdin:
    row = line.split(',')
    if count in range(1,52):
        verbal += int(row[2])
        math += int(row[3])
    count+=1
print "Total Verbal/Math SAT Scores of US States are:  {} / {}".format(verbal,math)
