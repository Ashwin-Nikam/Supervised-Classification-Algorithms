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
print(matrix)