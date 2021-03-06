import numpy as np
import itertools
import sys
import random
import math

"""
------------------------------------------------------------------------------------------------------------------------
"""


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


"""
------------------------------------------------------------------------------------------------------------------------
"""


def is_number(n):
    try:
        n = float(n)
    except:
        return False, None
    return True, n


"""
------------------------------------------------------------------------------------------------------------------------
"""


file = open("../Data/project3_dataset2.txt")
lines = file.readlines()
rows = len(lines)
columns = len(lines[0].split("\t"))
matrix = [[0 for x in range(columns)] for y in range(rows)]
for row in range(rows):
    for column in range(columns):
        matrix[row][column] = lines[row].split("\t")[column]
        matrix[row][column] = matrix[row][column].strip("\n")
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
        column = matrix[:, i]
        d = dict([(y, x) for x, y in enumerate(sorted(set(column)))])
        main_dictionary[i] = d
        mainArr.append("Categorical")

for i in range(len(mainArr)):
    if mainArr[i] == "Categorical" or mainArr[i] == "Class":
        d = main_dictionary[i]
        for j in range(len(matrix)):
            matrix[j][i] = d[matrix[j][i]]
matrix = matrix.astype(np.float)

"""
------------------------------------------------------------------------------------------------------------------------
"""


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


"""
------------------------------------------------------------------------------------------------------------------------
"""


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
    min = sys.maxsize
    if len(part_list) == 0:
        return
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
        if gini_a < min:
            min = gini_a
            split_value = split
        if len(unique) == 2:
            break
    split_values[column_index] = split_value
    gini_values[column_index] = min


"""
------------------------------------------------------------------------------------------------------------------------
"""


def handle_numerical_data(input_matrix, column_index, split_values, gini_values):
    split_value = 0
    min = sys.maxsize
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
        a = (len(split1) / rows) * gini1
        b = (len(split2) / rows) * gini2
        gini_a = a + b
        if gini_a < min:
            min = gini_a
            split_value = temp_matrix[row][column_index]
    split_values[column_index] = split_value
    gini_values[column_index] = min


"""
------------------------------------------------------------------------------------------------------------------------
"""


def compute_best_split(input_matrix, split_values, gini_values, column_list):
    random_features = random.sample(range(0, len(input_matrix[0])-1), m)
    for feature_index in random_features:
        if mainArr[feature_index] == "Numerical":
            handle_numerical_data(input_matrix, feature_index, split_values, gini_values)
        elif mainArr[feature_index] == "Categorical":
            if feature_index in column_list:
                continue
            else:
                handle_categorical_data(input_matrix, feature_index, split_values, gini_values)

    gini_values = np.array(gini_values)
    index = np.argmin(gini_values)
    criteria = split_values[index]
    return criteria, index


"""
------------------------------------------------------------------------------------------------------------------------
"""


def same_class(reduced_matrix):
    class_column = reduced_matrix[:, len(reduced_matrix[0])-1]
    unique = np.unique(class_column)
    if len(unique) > 1:
        return False, None
    return True, unique[0]


"""
------------------------------------------------------------------------------------------------------------------------
"""


def majority_class(reduced_matrix):
    class_column = reduced_matrix[:, len(reduced_matrix[0])-1]
    class_labels = np.unique(class_column)
    count1 = np.count_nonzero(class_column == class_labels[0])
    count2 = np.count_nonzero(class_column == class_labels[1])
    if count1 > count2:
        return class_labels[0]
    return class_labels[1]


"""
------------------------------------------------------------------------------------------------------------------------
"""


def split(criteria, column_index, input_matrix):
    left_set = []
    right_set = []
    if isinstance(criteria, list):
        for i in range(len(input_matrix)):
            value = input_matrix[i][column_index]
            if value in criteria:
                right_set.append(input_matrix[i])
            else:
                left_set.append(input_matrix[i])
    elif isinstance(criteria, float):
        for i in range(len(input_matrix)):
            value = input_matrix[i][column_index]
            if value >= criteria:
                right_set.append(input_matrix[i])
            else:
                left_set.append(input_matrix[i])
    return np.array(left_set), np.array(right_set)


"""
------------------------------------------------------------------------------------------------------------------------
"""


def height(root):
    if root is None:
        return -1
    else:
        return max(height(root.left), height(root.right)) + 1


"""
------------------------------------------------------------------------------------------------------------------------
"""


def traverse_tree(root, query):
    if root.final_value is not None:
        return root.final_value
    else:
        a = root.split_criteria
        if isinstance(a, list):
            if query[root.column_index] in a:
                return traverse_tree(root.right, query)
            else:
                return traverse_tree(root.left, query)
        elif isinstance(a, float):
            if query[root.column_index] >= a:
                return traverse_tree(root.right, query)
            else:
                return traverse_tree(root.left, query)


"""
------------------------------------------------------------------------------------------------------------------------
"""


def calculate_accuracy(class_list, test_data):
    test_data = matrix[test_data]
    class_label = test_data[:, len(test_data[0]) - 1]
    class_label = class_label.astype(np.int)
    class_list = np.array(class_list).astype(np.int)
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(class_label)):
        if class_list[i] == 1 and class_label[i] == 1:
            true_positive += 1
        elif class_list[i] == 0 and class_label[i] == 0:
            true_negative += 1
        elif class_list[i] == 0 and class_label[i] == 1:
            false_negative += 1
        elif class_list[i] == 1 and class_label[i] == 0:
            false_positive += 1
    accuracy = (true_positive + true_negative) / (true_positive + true_negative
                                                  + false_positive + false_negative)
    precision = (true_positive) / (true_positive + false_positive)
    recall = (true_positive) / (true_positive + false_negative)
    f1_measure = (2 * true_positive) / ((2 * true_positive) + false_positive + false_negative)
    return accuracy, precision, recall, f1_measure


"""
------------------------------------------------------------------------------------------------------------------------
"""


def calculate_each_test(root, test_data_idx):
    class_list = []
    test_data = matrix[test_data_idx]
    for i in range(len(test_data)):
        query = test_data[i]
        value = traverse_tree(root, query)
        class_list.append(value)
    return class_list


"""
------------------------------------------------------------------------------------------------------------------------
"""


def create_tree(records, old_list):
    col_vals = old_list.copy()
    flag, value = same_class(records)
    if flag:
        return Node(None, None, None, None, value)
    elif len(records) <= 3:                         # Check if node becomes too small (<= 3 examples)
        value = majority_class(records)
        return Node(None, None, None, None, value)
    else:
        if len(col_vals) < len(records[0]) - 1:     # Check if the height of tree exceeds no. of features.
            split_values = [sys.maxsize for i in range(len(records[0])-1)]
            gini_values = [sys.maxsize for i in range(len(records[0])-1)]
            criteria, column_index = compute_best_split(records, split_values, gini_values, col_vals)
            if criteria == sys.maxsize:
                value = majority_class(records)
                return Node(None, None, None, None, value)
            if mainArr[column_index] == "Categorical":
                col_vals.append(column_index)
            elif mainArr[column_index] == "Numerical":
                col_vals.append(sys.maxsize)
            node = Node(criteria, None, None, column_index, None)
            left_set, right_set = split(criteria, column_index, records)
            if len(left_set) == 0:
                value = majority_class(right_set)
                return Node(None, None, None, None, value)
            elif len(right_set) == 0:
                value = majority_class(left_set)
                return Node(None, None, None, None, value)
            else:
                node.left = create_tree(left_set, col_vals)
                node.right = create_tree(right_set, col_vals)
                return node
        else:
            value = majority_class(records)
            return Node(None, None, None, None, value)


"""
------------------------------------------------------------------------------------------------------------------------
"""

number_of_trees = 5     # Number of trees in a forest
m = round(math.sqrt(len(matrix[0])-1))
print("m = ", m)
print("t = ", number_of_trees)
print()
folds = 10              # Number of forests.

part_len = int(len(matrix) / folds)
metrics_avg = [0.0, 0.0, 0.0, 0.0]
train_data_idx = set()
accuracy_list = []
precision_list = []
recall_list = []
f1_measure_list = []
for i in range(folds):
    print("Fold ", i + 1)
    if i != folds - 1:
        start = (i * part_len)
        end = start + part_len
        test_data_idx = set(range(start, end))
    else:
        test_data_idx = set(range(i * part_len, len(matrix)))
    train_data_idx = set(range(len(matrix))).difference(test_data_idx)
    test_data_idx = list(test_data_idx)
    train_data_idx = list(train_data_idx)
    train_data = matrix[train_data_idx]     # Train data which needs to be sampled 10 times
    test_data = matrix[test_data_idx]       # Fixed test data

    root_list = []
    main_class_list = []
    sample_train_data_idx = []

    """
    Bagging to get n samples from the fixed data where n is the
    number of trees to be generated in a forest.
    """

    for j in range(number_of_trees):
        sample_train_data_idx = np.random.choice(train_data_idx, len(train_data_idx),
                                                 replace=True)
        sample_train_data = matrix[sample_train_data_idx]
        root_list.append(create_tree(sample_train_data, []))

    """
    Calculate accuracy of each tree in the forest.
    """

    for root in root_list:
        class_list = calculate_each_test(root, test_data_idx)
        main_class_list.append(class_list)

    """
    Calculate accuracy of the entire forest.
    This is done by performing voting.
    """

    main_class_list = np.array(main_class_list, dtype=np.float64)
    final_class_list = []
    for i in range(len(main_class_list[0])):
        class_column = main_class_list[:, i]
        unique = np.unique(class_column)
        if len(unique) == 1:
            final_class_list.append(unique[0])
        else:
            count1 = np.count_nonzero(class_column == unique[0])
            count2 = np.count_nonzero(class_column == unique[1])
            if count1 > count2:
                final_class_list.append(unique[0])
            else:
                final_class_list.append(unique[1])
    accuracy, precision, recall, f1_measure = calculate_accuracy(final_class_list, test_data_idx)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_measure_list.append(f1_measure)
    print("Accuracy :", accuracy)
    print("Precision :", precision)
    print("Recall :", recall)
    print("F1-Measure :", f1_measure)
    print()

"""
Calculate average accuracy of the 10 folds or 
10 forests in this case. Here each forest has 
computed accuracy for different test data.
"""

print()
print("********** Final answer ************")
accuracy = np.sum(accuracy_list)/len(accuracy_list)
precision = np.sum(precision_list)/len(precision_list)
recall = np.sum(recall_list)/len(recall_list)
f1_measure = np.sum(f1_measure_list)/len(f1_measure_list)
print("Average Accuracy: ", accuracy, "\nAverage Precision: ", precision, "\nAverage Recall: ", recall,
"\nAverage F1-measure: ", f1_measure)


"""
------------------------------------------------------------------------------------------------------------------------
"""