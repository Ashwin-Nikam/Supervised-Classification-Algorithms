import numpy as np
import pandas as pd

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
matrix = np.delete(matrix, columns-1, 1) #matrix contains all the data
columns = columns-1

def calculate_impurity(split_matrix):
    

for i in range(columns):
    temp_matrix = matrix.copy()
    temp_matrix = temp_matrix[temp_matrix[:,i].argsort()]
    column = temp_matrix[:,i]
    for row in range(rows):
        index1 = list(range(0,row))
        index2 = list(range(row, rows))
        split1 = temp_matrix[index1]
        split2 = temp_matrix[index2]

