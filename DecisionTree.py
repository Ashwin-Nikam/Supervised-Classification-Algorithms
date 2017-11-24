import numpy as np
import pandas as pd
import sys

file = open("project3_dataset2.txt")
lines = file.readlines()
rows = len(lines)
columns = len(lines[0].split("\t"))
matrix = [[0 for x in range(columns)] for y in range(rows)]
for row in range(rows):
    for column in range(columns):
        matrix[row][column] = lines[row].split("\t")[column]
matrix = np.array(matrix)
true_values = np.array(matrix)[:,columns-1] #true_values contains the true labels
mainArr = []


def is_number(n):
    try:
        n = float(n)
    except:
        return False, None
    return True, n


for i in range(len(matrix[0])):
    status, number = is_number(matrix[0][i])
    if i == len(matrix[0])-1:
        mainArr.append("Class")
    elif status:
        mainArr.append("Numerical")
    else:
        mainArr.append("Categorical")


def calculate_gini(split_matrix):
    den = len(split_matrix)
    if den == 0:
        return 0
    num0 = 0
    num1 = 0
    for i in range(den):
        split_matrix[i][columns - 1] = split_matrix[i][columns-1].rstrip("\n")
        if split_matrix[i][columns-1] == '0':
            num0 += 1
        elif split_matrix[i][columns-1] == '1':
            num1 += 1
    probability0 = num0/den
    probability1 = num1/den
    gini = 1 - (probability0**2) - (probability1**2)
    return gini


def handle_categorical_data():
    return


main_gini = calculate_gini(matrix)
max = -sys.maxsize
split_points = []
for i in range(columns-1):
    if mainArr[i] == "Categorical":
        handle_categorical_data()
    else:
        split_point = 0
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
                split_point = row
        split_points.append(split_point)
        print("Max ", max)
        print("Split point ",split_point)