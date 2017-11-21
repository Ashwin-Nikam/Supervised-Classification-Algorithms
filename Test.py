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

mainArr = []
for word in matrix[0]:
    status, number = is_number(word)
    if status:
        mainArr.append("Numerical")
    else:
        mainArr.append("Categorical")

meanVector = []
varianceVector = []
for i in range(len(mainArr)-1):
    if mainArr[i] == "Numerical":
        meanVector.append(np.mean(matrix[:,i].astype(np.float)))
        varianceVector.append(np.var(matrix[:,i].astype(np.float)))
    elif mainArr[i] == "Categorical":
        meanVector.append("Categorical")
        varianceVector.append("Categorical")
#print(meanVector)
#print(varianceVector)

test = [17, 15, 23, 7, 9, 13]
def calculate_mean_and_variance(column):
    sum = 0
    newCol = []
    mean = np.mean(column)
    for i in range(len(column)):
        newCol.append(column[i]-mean)
        newCol[i] = newCol[i]**2
        sum += newCol[i]
    return np.mean(column), sum/(len(newCol)-1)
print(calculate_mean_and_variance(test))