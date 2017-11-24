import numpy as np
import itertools
import sys


class Node:
    split_criteria = None
    left = None
    right = None

    def __init__(self, criteria, left, right):
        self.split_criteria = criteria
        self.left = left
        self.right = right


def is_number(n):
    try:
        n = float(n)
    except:
        return False, None
    return True, n


file = open("new_dataset.txt")
lines = file.readlines()
rows = len(lines)
columns = len(lines[0].split("\t"))
matrix = [[0 for x in range(columns)] for y in range(rows)]
for row in range(rows):
    for column in range(columns):
        matrix[row][column] = lines[row].split("\t")[column]
        status, number = is_number(matrix[row][column])
        if status:
            matrix[row][column] = number
matrix = np.array(matrix)
true_values = np.array(matrix)[:,columns-1] #true_values contains the true labels
mainArr = []
main_dictionary = {}

for i in range(len(matrix[0])):
    status, number = is_number(matrix[0][i])
    if i == len(matrix[0])-1:
        column = matrix[:, i]
        d = dict([(y, x) for x, y in enumerate(sorted(set(column)))])
        main_dictionary[i] = d
        mainArr.append("Class")
    elif status:
        mainArr.append("Numerical")
    else:
        column = matrix[:,i]
        d = dict([(y, x) for x, y in enumerate(sorted(set(column)))])
        main_dictionary[i] = d
        mainArr.append("Categorical")

for i in range(len(mainArr)):
    if mainArr[i] == "Categorical" or mainArr[i] == "Class":
        d = main_dictionary[i]
        for j in range(len(matrix)):
            matrix[j][i] = d[matrix[j][i]]
matrix = matrix.astype(np.float)


def calculate_gini(split_matrix):
    den = len(split_matrix)
    if den == 0:
        return 0
    num0 = 0
    num1 = 0
    for i in range(den):
        if split_matrix[i][columns-1] == 0:
            num0 += 1
        elif split_matrix[i][columns-1] == 1:
            num1 += 1
    probability0 = num0/den
    probability1 = num1/den
    gini = 1 - (probability0**2) - (probability1**2)
    return gini


split_values = []
gin_values = []
main_gini = calculate_gini(matrix)
max = -sys.maxsize


def handle_categorical_data(column_index):
    column = matrix[:,column_index]
    unique = np.unique(column)
    part_list = []
    for i in range(len(unique)):
         partitions = itertools.combinations(unique, i)
         for j in partitions:
             if (len(j) > 0):
                 part_list.append(list(j))
    split_value = 0
    max = -sys.maxsize
    for split in part_list:
        split1 = []
        split2 = []
        for i in range(len(matrix)):
            count = 0
            for j in split:
                if matrix[i][column_index] == j:
                    count += 1
                    split1.append(matrix[i])
            if count == 0:
                split2.append(matrix[i])
        split1 = np.array(split1)
        split2 = np.array(split2)
        gini1 = calculate_gini(split1)
        gini2 = calculate_gini(split2)
        a = (len(split1) / rows) * gini1
        b = (len(split2) / rows) * gini2
        gini_a = a + b
        diff = main_gini - gini_a
        if diff > max:
            max = diff
            split_value = split
    split_values.append(split_value)
    gin_values.append(max)


for i in range(columns-1):
    if mainArr[i] == "Categorical":
        handle_categorical_data(i)
    else:
        split_value = 0
        max = -sys.maxsize
        temp_matrix = matrix.copy()
        temp_matrix = temp_matrix[temp_matrix[:,i].argsort()]
        for row in range(rows):
            index1 = list(range(0,row))
            index2 = list(range(row, rows))
            split1 = temp_matrix[index1]
            split2 = temp_matrix[index2]
            gini1 = calculate_gini(split1)
            gini2 = calculate_gini(split2)
            a = (len(index1)/rows)*gini1
            b = (len(index2)/rows)*gini2
            gini_a = a + b
            diff = main_gini - gini_a
            if diff > max:
                max = diff
                split_value = temp_matrix[row][i]
        split_values.append(split_value)
        gin_values.append(max)
print(split_values)
#print(gin_values)
print(matrix)


def same_class(reduced_matrix):
    value = reduced_matrix[0][len(reduced_matrix[0])-1]
    for i in range(len(reduced_matrix)):
        if reduced_matrix[i][len(reduced_matrix[i])-1] != value:
            return False
    return True


def majority_class(reduced_matrix):
    count1 = 0
    count2 = 0
    class_column = reduced_matrix[:,len(reduced_matrix[0])-1]
    class_labels = np.unique(class_column)
    for label in class_column:
        if label == class_labels[0]:
            count1 += 1
        elif label == class_labels[1]:
            count2 += 1
    if count1 > count2:
        return class_labels[0]
    return class_labels[1]


def split(criteria, column_index, input_matrix):
    left_set = []
    right_set = []
    if isinstance(criteria, list):      #Categorical
        for i in range(len(input_matrix)):
            value = input_matrix[i][column_index]
            if value in criteria:
                right_set.append(input_matrix[i])
            else:
                left_set.append(input_matrix[i])
    elif isinstance(criteria, float):   #Numerical
        for i in range(len(input_matrix)):
            value = input_matrix[i][column_index]
            if value >= criteria:
                right_set.append(input_matrix[i])
            else:
                left_set.append(input_matrix[i])
    return np.array(left_set), np.array(right_set)



"""
def mainMethod(records)
    status = same_class(records)
    if yes:
        return Node(class)
    else:
        if attribute left:
            criteria = computeBestSplit(records)  //remember to not take this attribute again        
            Node node = new Node("criteria");
            left_set, right_set = split(criteria, records)
            node.left = mainMethod(left_set)
            node.right = mainMethod(right_set)
            return node
        else:
            class = majorityClass(records)
            return new Node("class)

root = mainMethod(matrix)
"""