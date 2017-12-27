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


file = open("../Data/project3_dataset4.txt")
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
    if i == len(matrix[0]) - 1:
        mainArr.append("Class")
    elif status:
        mainArr.append("Numerical")
    else:
        mainArr.append("Categorical")


"""
------------------------------------------------------------------------------------------------------------------------
"""


def prior_probability(classLabel):
    column = matrix[:, len(matrix[0]) - 1]
    l = list(column)
    num = l.count(classLabel)
    den = len(column)
    return num / den


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
    den = l.count(
        classLabel)  # for a categorical value, we return the number of times that value has appeared for the given class label divided by the total count of that class label
    return num / den


"""
------------------------------------------------------------------------------------------------------------------------
"""


def mean_var_in_dict(input_matrix):
    num_classes = np.unique(matrix[:, len(matrix[
                                              0]) - 1]).size  # get the total class labels . Int this case our count is 2 since we have only 0 and 1 as labels
    mean_std_dict[0] = []
    mean_std_dict[1] = []
    matrix_0 = []
    matrix_1 = []
    for i in range(len(input_matrix)):
        if input_matrix[i][len(input_matrix[0]) - 1] == '0':
            matrix_0.append(input_matrix[i])
        elif input_matrix[i][len(input_matrix[0]) - 1] == '1':
            matrix_1.append(input_matrix[i])
    matrix_0 = np.array(
        matrix_0)  # matrix0 is a list of rows having class label 0s and matrix1 is a list of rows having class labels 1s
    matrix_1 = np.array(matrix_1)

    for i in range(len(mainArr) - 1):  # for each column of the row except the class label column
        for j in range(num_classes):  # for each class label
            if mainArr[i] == "Numerical":  # if the column has numerical values then
                temp = []
                if j == 0:  # append the mean and the standard deviation of that column to a list and append this list to dictionary
                    main_col = matrix_0[:, i]
                    main_col = main_col.astype(np.float)
                    mean = np.mean(main_col)
                    std = np.std(main_col, ddof=1)
                    temp.append(mean)
                    temp.append(std)
                    mean_std_dict[j].append(temp)  # for 0 class label append it to the 0th index of dictionary
                else:
                    main_col = matrix_1[:, i]
                    main_col = main_col.astype(np.float)
                    mean = np.mean(main_col)
                    std = np.std(main_col, ddof=1)
                    temp.append(mean)
                    temp.append(std)
                    mean_std_dict[j].append(temp)  # for class label 1 append it to the 1st index of dictionary
            elif mainArr[i] == "Categorical":
                mean_std_dict[j].append(
                    ["Categorical"])  # incase of categorical data append categorical to the dictionary


"""
------------------------------------------------------------------------------------------------------------------------
"""


def calculate_posterior_probability(test_data, train_data):
    main_list = []
    for row in test_data:  # for every row of test data
        query = list(row)
        query.pop()  # remove the last column from the list of columns in the row
        numClasses = np.unique(matrix[:, len(matrix[0]) - 1]).size  # class label  count
        finalList = []
        for i in range(numClasses):  # for each class label
            probability = 1.0
            for j in range(len(query)):  # for each column
                if mainArr[j] == "Numerical":
                    mean_std_list = mean_std_dict.get(i)[j]  # get the list corresponding to the column from dictionary
                    mu = mean_std_list[0]
                    sigma = mean_std_list[1]
                    x = query[j]  # calculate pdf for each column of that row
                    probability *= ss.norm(mu, sigma).pdf(float(x))
                else:
                    probability *= categorical_probability(query[j], j, str(i))
            prior = prior_probability(str(i))
            finalList.append(prior * probability)  # prior probability * descriptor posterior probability
        main_list.append(finalList.index(np.amax(finalList)))
    return main_list


"""
------------------------------------------------------------------------------------------------------------------------
"""


def calculate_accuracy(class_list, test_data):
    test_data = matrix[test_data]
    class_label = test_data[:, len(test_data[0]) - 1]
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
    precision_denominator = (true_positive + false_positive)
    if precision_denominator == 0:  # In case if denominator becomes 0 for precision or recall or f1_measure, in that case return 0 for their values
        precision = 0
    else:
        precision = (true_positive) / precision_denominator
    recall_denominator = (true_positive + false_negative)
    if recall_denominator == 0:
        recall = 0
    else:
        recall = (true_positive) / recall_denominator
    f1_measure_denominator = ((2 * true_positive) + false_positive + false_negative)
    if f1_measure_denominator == 0:
        f1_measure = 0
    else:
        f1_measure = (2 * true_positive) / f1_measure_denominator
    return accuracy, precision, recall, f1_measure


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
    return answer


"""
------------------------------------------------------------------------------------------------------------------------
"""

mean_var_in_dict(matrix)
query = input("Enter query: ")
query = query.split('/')
if len(query) == len(matrix[0]) - 1:
    numClasses = np.unique(matrix[:, len(matrix[0]) - 1]).size  # class label  count
    finalList = []
    main_list = []
    for i in range(numClasses):  # for each class label
        probability = 1.0
        for j in range(len(query)):  # for each column
            if mainArr[j] == "Numerical":
                mean_std_list = mean_std_dict.get(i)[j]  # get the list corresponding to that class label
                mu = mean_std_list[0]
                sigma = mean_std_list[1]
                x = query[j]  # calculate pdf for each column of that row
                probability *= ss.norm(mu, sigma).pdf(float(x))
            else:
                multiple = categorical_probability(query[j], j, str(i))
                probability *= multiple
        prior = prior_probability(str(i))
        descriptor = calculate_descriptor_prior(query, matrix)
        finalList.append((prior * probability)/descriptor)  # prior probability * descriptor posterior probability
    print("Class 0 Probability P( H0 | X ): ", finalList[0])
    print("Class 1 Probability P( H1 | X ): ", finalList[1])
    main_list.append(finalList.index(np.amax(finalList)))
    print("Final Class: ", main_list[0])
else:
    print("Invalid input. Please enter again. ")

"""
------------------------------------------------------------------------------------------------------------------------
"""