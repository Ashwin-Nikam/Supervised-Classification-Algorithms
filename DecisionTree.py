import numpy as np
import itertools
import sys
from queue import *


class Node(object):
    split_criteria = None
    left = None
    right = None
    column_index = None
    final_value = None

    def __init__(self, criteria, left, right, column_index, final_value):
        self.split_criteria = criteria
        self.left = left
        self.right = right
        self.column_index = column_index
        self.final_value = final_value


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


main_gini = calculate_gini(matrix)


def handle_categorical_data(input_matrix, column_index, split_values, gini_values):
    rows = len(input_matrix)
    column = input_matrix[:,column_index]
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
        for i in range(len(input_matrix)):
            count = 0
            for j in split:
                if input_matrix[i][column_index] == j:
                    count += 1
                    split1.append(input_matrix[i])
            if count == 0:
                split2.append(input_matrix[i])
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
    gini_values.append(max)


def handle_numerical_data(input_matrix, column_index, split_values, gini_values):
    split_value = 0
    max = -sys.maxsize
    temp_matrix = input_matrix.copy()
    temp_matrix = temp_matrix[temp_matrix[:, column_index].argsort()]
    rows = len(input_matrix)
    for row in range(rows):
        index1 = list(range(0, row))
        index2 = list(range(row, rows))
        split1 = temp_matrix[index1]
        split2 = temp_matrix[index2]
        gini1 = calculate_gini(split1)
        gini2 = calculate_gini(split2)
        a = (len(index1) / rows) * gini1
        b = (len(index2) / rows) * gini2
        gini_a = a + b
        diff = main_gini - gini_a
        if diff > max:
            max = diff
            split_value = temp_matrix[row][column_index]
    split_values.append(split_value)
    gini_values.append(max)


def compute_best_split(input_matrix, split_values, gini_values, column_list):
    for i in range(len(input_matrix[0])-1):
        if i in column_list:
            split_values.append(-sys.maxsize)
            gini_values.append(-sys.maxsize)
        if mainArr[i] == "Categorical":
            handle_categorical_data(input_matrix, i, split_values, gini_values)
        elif mainArr[i] == "Numerical":
            handle_numerical_data(input_matrix, i, split_values, gini_values)

    gini_values = np.array(gini_values)
    index = np.argmax(gini_values)
    criteria = split_values[index]
    return criteria, index


def same_class(reduced_matrix):
    value = reduced_matrix[0][len(reduced_matrix[0])-1]
    for i in range(len(reduced_matrix)):
        if reduced_matrix[i][len(reduced_matrix[i])-1] != value:
            return False, None
    return True, value


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


def main_method(records, old_list):
    if len(records) == 0:
        return None
    col_vals = old_list.copy()
    flag, value = same_class(records)
    if flag:
        return Node(None, None, None, None, value)
    else:
        if len(col_vals) < len(records[0])-2:
            split_values = []
            gini_values = []
            criteria, column_index = compute_best_split(records, split_values, gini_values, col_vals)
            col_vals.append(column_index)
            node = Node(criteria, None, None, column_index, None)
            left_set, right_set = split(criteria, column_index, records)
            node.left = main_method(left_set, col_vals)
            node.right = main_method(right_set, col_vals)
            return node
        else:
            value = majority_class(records)
            return Node(None, None, None, None, value)

column_list = []
root = main_method(matrix, column_list)


def print_tree():
    q = Queue(maxsize=0)
    q.put(root)
    while not q.empty():
        count = q.qsize()
        for i in range(count):
            node = q.get()
            if node.split_criteria != None:
                print(node.split_criteria)
            else:
                print(node.final_value,"!")
            if node.left is not None:
                q.put(node.left)
            if node.right is not None:
                q.put(node.right)
        print("=======")


def traverse_tree(root, query):
    if root.final_value is not None:
        print("!!!!!",root.final_value)
        return
    else:
        a = root.split_criteria
        if isinstance(a, list):
            if query[root.column_index] in a:
                traverse_tree(root.right, query)
            else:
                traverse_tree(root.left, query)
        elif isinstance(a, float):
            if query[root.column_index] >= a:
                traverse_tree(root.right, query)
            else:
                traverse_tree(root.left, query)


query = matrix[10]
traverse_tree(root, query)