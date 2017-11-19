import numpy as np

def is_number(n):
    try:
        number = float(n)
    except:
        return False, None
    return True, number

f = open('project3_dataset2.txt', newline='\n')
matrix = []
for line in f:
    line = line.split('\t')
    entry = []
    for word in line:
        if word == line[len(line) - 1]:
            word = word.rstrip("\n")  # remove trailing \n
        status, number = is_number(word)
        if status:
            entry.append(number)
        else:
            entry.append(word)  # nominal attribute. Store unconverted
    matrix.append(entry)

matrix = np.array(matrix)
#print(matrix)

mainArr = []
for word in matrix[0]:
    status, number = is_number(word)
    if status:
        mainArr.append("Numerical")
    else:
        mainArr.append("Categorical")
print(mainArr)

"""
k = 10
myColumn = matrix[:,0]
myColumn = np.array(myColumn)
myColumn = myColumn.astype(np.float)
max = np.amax(myColumn)
min = np.amin(myColumn)
size = (max - min)/k
bins = []
temp = min
for i in range(k):
    bins.append(temp)
    temp = temp + size
bins = np.array(bins)
binning = np.digitize(myColumn, bins)
print(binning[291])
"""