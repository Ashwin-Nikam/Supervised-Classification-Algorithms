import numpy as np
import scipy.stats as ss

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


file = open("project3_dataset2.txt")
lines = file.readlines()
rows = len(lines)
columns = len(lines[0].split("\t"))
matrix = [[0 for x in range(columns)] for y in range(rows)]
for row in range(rows):
    for column in range(columns):
        matrix[row][column] = lines[row].split("\t")[column]
        matrix[row][column] = matrix[row][column].rstrip("\n")
matrix = np.array(matrix)
mean_std_dict = {}
mainArr = []

"""
------------------------------------------------------------------------------------------------------------------------
"""


for i in range(len(matrix[0])):
    status, number = is_number(matrix[0][i])
    if i == len(matrix[0])-1:
        mainArr.append("Class")
    elif status:
        mainArr.append("Numerical")
    else:
        mainArr.append("Categorical")


"""
------------------------------------------------------------------------------------------------------------------------
"""


def prior_probability(classLabel):
    column = matrix[:,len(matrix[0])-1]
    l = list(column)
    matrix[:, len(matrix[0]) - 1] = column
    num = l.count(classLabel)
    den = len(column)
    return num/den


"""
------------------------------------------------------------------------------------------------------------------------
"""


def categorical_probability(category, colIndex, classLabel):
    num = 0
    for i in range(len(matrix)):
        if matrix[i][colIndex] == category and matrix[i][len(matrix[0]) - 1] == classLabel:
            num += 1
    column = matrix[:, len(matrix[0]) - 1]
    l = list(column)
    den = l.count(classLabel)
    return num/den


"""
------------------------------------------------------------------------------------------------------------------------
"""


def mean_var_in_dict(input_matrix):
    mean_std_dict[0] = []
    mean_std_dict[1] = []
    matrix_0 = []
    matrix_1 = []
    for i in range(len(input_matrix)):
        if input_matrix[i][len(input_matrix[0]) - 1] == '0':
            matrix_0.append(input_matrix[i])
        elif input_matrix[i][len(input_matrix[0]) - 1] == '1':
            matrix_1.append(input_matrix[i])
    matrix_0 = np.array(matrix_0)
    matrix_1 = np.array(matrix_1)

    for i in range(len(mainArr)-1):
        for j in range(2):
            if mainArr[i] == "Numerical":
                temp = []
                if j == 0:
                    main_col = matrix_0[:,i]
                    main_col = main_col.astype(np.float)
                    mean = np.mean(main_col)
                    std = np.std(main_col, ddof=1)
                    temp.append(mean)
                    temp.append(std)
                    mean_std_dict[j].append(temp)
                else:
                    main_col = matrix_1[:, i]
                    main_col = main_col.astype(np.float)
                    mean = np.mean(main_col)
                    std = np.std(main_col, ddof=1)
                    temp.append(mean)
                    temp.append(std)
                    mean_std_dict[j].append(temp)
            elif mainArr[i] == "Categorical":
                mean_std_dict[j].append(["Categorical"])


"""
------------------------------------------------------------------------------------------------------------------------
"""


def calculate_descriptor_prior(query, input_matrix):
    answer = 1.0
    for j in range(len(query)):
        column = input_matrix[:, j]
        l = list(column)
        num = l.count(query[j])
        den = len(l)
        answer *= (num/den)
    if answer == 0:
        print("")
    return answer


"""
------------------------------------------------------------------------------------------------------------------------
"""


def calculate_posterior_probability(test_data, train_data):
    main_list = []
    for row in test_data:
        query = list(row)
        query.pop()
        finalList = []
        for i in range(2):
            probability = 1.0
            for j in range(len(query)):
                if mainArr[j] == "Numerical":
                    mean_std_list = mean_std_dict.get(i)[j]
                    mu = mean_std_list[0]
                    sigma = mean_std_list[1]
                    x = query[j]
                    probability *= ss.norm(mu, sigma).pdf(float(x))
                else:
                    probability *= categorical_probability(query[j], j, str(i))
            prior = prior_probability(str(i))
            finalList.append(prior * probability)
        main_list.append(finalList.index(np.amax(finalList)))
    return main_list


"""
------------------------------------------------------------------------------------------------------------------------
"""


def calculate_accuracy(class_list, test_data):
    test_data = matrix[test_data]
    class_label = test_data[:,len(test_data[0])-1]
    class_label = class_label.astype(np.int)
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
    if true_negative == 0 or false_negative == 0 or false_positive == 0:
        precision = 0
        recall = 0
        f1_measure = 0
        return accuracy, precision, recall, f1_measure
    else:
        precision = (true_positive) / (true_positive + false_positive)
        recall = (true_positive) / (true_positive + false_negative)
        f1_measure = (2 * true_positive) / ((2 * true_positive) + false_positive + false_negative)
        return accuracy, precision, recall, f1_measure


"""
------------------------------------------------------------------------------------------------------------------------
"""

folds = 10
part_len = int(len(matrix) / folds)
metrics_avg = [0.0,0.0,0.0,0.0]
train_data_idx = set()
accuracy_list = []
precision_list = []
recall_list = []
f1_measure_list = []
for i in range(folds):
    if i != folds - 1:
        start = (i * part_len)
        end = start + part_len
        test_data_idx = set(range(start, end))
    else:
        test_data_idx = set(range(i * part_len, len(matrix)))
    train_data_idx = set(range(len(matrix))).difference(test_data_idx)
    test_data_idx = list(test_data_idx)
    train_data_idx = list(train_data_idx)
    test_data_idx.sort()
    train_data_idx.sort()

    train_data = matrix[train_data_idx]
    test_data= matrix[test_data_idx]
    mean_var_in_dict(train_data)  #Updating dictionary with mean and variance for new train data
    class_list = calculate_posterior_probability(test_data, train_data) #Calculating probability for every row in test data

    accuracy, precision, recall, f1_measure = calculate_accuracy(class_list, test_data_idx)
    print("Fold: ", i+1)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_measure_list.append(f1_measure)
    print("Accuracy :", accuracy)
    print("Precision :", precision)
    print("Recall :", recall)
    print("F1-Measure :", f1_measure)
    print()

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