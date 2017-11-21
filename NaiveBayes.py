import numpy as np
import scipy.stats as ss

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


def prior_probability(classLabel):
    column = matrix[:,len(matrix[0])-1]
    for i in range(len(column)):
        column[i] = column[i].strip("\n")
    l = list(column)
    matrix[:, len(matrix[0]) - 1] = column
    num = l.count(classLabel)
    den = len(column)
    return num/den


def calculate_mean_and_variance(column, classValue):
    column = column.astype(np.float)
    sum = 0
    newCol = []
    count = 0
    for i in range(len(column)):
        if int(matrix[i][len(matrix[0])-1]) == classValue:
            sum += column[i]
            count += 1
    mean = sum/count
    sum = 0
    for i in range(len(column)):
        if int(matrix[i][len(matrix[0]) - 1]) == classValue:
            temp = column[i]-mean
            sum += temp**2
            newCol.append(temp)
    return mean, sum/(len(newCol)-1)


def categorical_probability(category, colIndex, classLabel):
    num = 0
    for i in range(len(matrix)):
        matrix[i][len(matrix[0]) - 1] = matrix[i][len(matrix[0]) - 1].rstrip("\n")
        if matrix[i][colIndex] == category and np.equal(matrix[i][len(matrix[0]) - 1], classLabel):
            num += 1
    column = matrix[:, len(matrix[0]) - 1]
    l = list(column)
    den = l.count(classLabel)
    return num/den


numClasses = np.unique(matrix[:, len(matrix[0])-1]).size
mean_var_dict = {}
list0 = []
list1 = []
mean_var_dict[0] = list0
mean_var_dict[1] = list1
for i in range(len(mainArr)-1):
    for j in range(numClasses):
        if mainArr[i] == "Numerical":
            mean, var = calculate_mean_and_variance(matrix[:,i], j)
            temp = []
            if j is 0:
                temp.append(mean)
                temp.append(var)
                mean_var_dict[j].append(temp)
            else:
                temp.append(mean)
                temp.append(var)
                mean_var_dict[j].append(temp)
        elif mainArr[i] == "Categorical":
            mean_var_dict[j].append(["Categorical"])


k = 0
query = list(matrix[k])
query.pop()
numClasses = np.unique(matrix[:, len(matrix[0])-1]).size
finalList = []
for i in range(numClasses):
    probability = 1
    for j in range(len(query)):
        if(mainArr[j] == "Numerical"):
            mean_var_list = mean_var_dict.get(i)[j]
            mu = mean_var_list[0]
            var = mean_var_list[1]
            x = query[j]
            sigma = var**(1/2)
            probability *= ss.norm(mu, sigma).pdf(float(x))
        else:
            probability *= categorical_probability(query[j], j, str(i))
    finalList.append(probability)
print(np.amax(finalList), " ", finalList.index(np.amax(finalList)))