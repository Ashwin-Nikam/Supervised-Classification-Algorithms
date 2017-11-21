import numpy as np


def is_number(n):
    try:
        number = float(n)
    except:
        return False, None
    return True, number


def binning(column):
    k = 10
    column = np.array(column)
    column = column.astype(np.float)
    max = np.amax(column)
    min = np.amin(column)
    size = (max - min) / k
    bins = []
    temp = min
    for i in range(k):
        bins.append(temp)
        temp = temp + size
    bins = np.array(bins)
    binning = np.digitize(column, bins)
    return binning


file = open("project3_dataset2.txt")
lines = file.readlines()
rows = len(lines)
columns = len(lines[0].split("\t"))
matrix = [[0 for x in range(columns)] for y in range(rows)]
for row in range(rows):
    for column in range(columns):
        matrix[row][column] = lines[row].split("\t")[column]
matrix = np.array(matrix)

mainArr = []
for i in range(len(matrix[0])):
    status, number = is_number(matrix[0][i])
    if i == len(matrix[0])-1:
        mainArr.append("Class")
    elif status:
        mainArr.append("Numerical")
    else:
        mainArr.append("Categorical")

# for i in range(len(matrix[0])):
#     if mainArr[i] == "Categorical" or mainArr[i] == "Class":
#         continue
#     column = matrix[:,i]
#     matrix[:,i] = binning(column)


def prior_probability(classLabel):
    column = matrix[:,len(matrix[0])-1]
    for i in range(len(column)):
        column[i] = column[i].strip("\n")
    l = list(column)
    matrix[:, len(matrix[0]) - 1] = column
    num = l.count(classLabel)
    den = len(column)
    return num/den


def calculate_mean_and_variance(column):
    column = column.astype(np.float)
    sum = 0
    newCol = []
    mean = np.mean(column)
    for i in range(len(column)):
        newCol.append(column[i]-mean)
        newCol[i] = newCol[i]**2
        sum += newCol[i]
    return np.mean(column), sum/(len(newCol)-1)


def dtr_probability(query, classLabel):
    answer = 1.0
    for j in range(len(query)):
        num = 0
        for i in range(len(matrix)):
            if matrix[i][j] == query[j] and np.equal(matrix[i][len(matrix[0])-1], classLabel):
                num += 1
        column = matrix[:, len(matrix[0]) - 1]
        l = list(column)
        den = l.count(classLabel)
        answer *= float(num+1)/float(den) #Here we add num+1 for avoiding zeroes
    return answer


meanVector = []
varianceVector = []
for i in range(len(mainArr)-1):
    if mainArr[i] == "Numerical":
        mean, var = calculate_mean_and_variance(matrix[:,i])
        meanVector.append(mean)
        varianceVector.append(var)
    elif mainArr[i] == "Categorical":
        meanVector.append("Categorical")
        varianceVector.append("Categorical")

print(meanVector)
print(varianceVector)

k = 1
query = list(matrix[k])
query.pop()
numClasses = np.unique(matrix[:, len(matrix[0])-1]).size
finalList = []
for i in range(numClasses):
    finalList.append(prior_probability(str(i)) * dtr_probability(query, str(i)))
#print(np.amax(finalList), " ", finalList.index(np.amax(finalList)))