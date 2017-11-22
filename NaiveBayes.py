import numpy as np
import scipy.stats as ss


def is_number(n):
    try:
        n = float(n)
    except:
        return False, None
    return True, n


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


mean_var_dict = {}
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
    l = list(column)
    matrix[:, len(matrix[0]) - 1] = column
    num = l.count(classLabel)
    den = len(column)
    return num/den


def calculate_mean_and_variance(column, input_matrix, classValue):
    column = column.astype(np.float)
    sum = 0
    newCol = []
    count = 0
    for i in range(len(column)):
        if int(input_matrix[i][len(matrix[0])-1]) == classValue:
            sum += column[i]
            count += 1
    mean = sum/count
    sum = 0
    for i in range(len(column)):
        if int(input_matrix[i][len(matrix[0]) - 1]) == classValue:
            temp = column[i]-mean
            sum += temp**2
            newCol.append(temp)
    return mean, sum/(len(newCol)-1)


def categorical_probability(category, colIndex, classLabel):
    num = 0
    for i in range(len(matrix)):
        if matrix[i][colIndex] == category and np.equal(matrix[i][len(matrix[0]) - 1], classLabel):
            num += 1
    column = matrix[:, len(matrix[0]) - 1]
    l = list(column)
    den = l.count(classLabel)
    return num/den


def mean_var_in_dict(input_matrix):
    num_classes = np.unique(matrix[:, len(matrix[0])-1]).size
    list0 = []
    list1 = []
    mean_var_dict[0] = list0
    mean_var_dict[1] = list1
    for i in range(len(mainArr)-1):
        for j in range(num_classes):
            if mainArr[i] == "Numerical":
                mean, var = calculate_mean_and_variance(input_matrix[:,i], input_matrix, j)
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


def calculate_posterior_probability(test_data, train_data):
    main_list = []
    for row in test_data:
        query = list(row)
        query.pop()
        numClasses = np.unique(matrix[:, len(matrix[0])-1]).size
        finalList = []
        for i in range(numClasses):
            probability = 1
            for j in range(len(query)):
                if mainArr[j] == "Numerical":
                    mean_var_list = mean_var_dict.get(i)[j]
                    mu = mean_var_list[0]
                    var = mean_var_list[1]
                    x = query[j]
                    sigma = var**(1/2)
                    probability *= ss.norm(mu, sigma).pdf(float(x))
                else:
                    probability *= categorical_probability(query[j], j, str(i))
            prior = prior_probability(str(i))
            descriptor = probability
            finalList.append(prior * descriptor)
        main_list.append(finalList.index(np.amax(finalList)))
    return main_list


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
    precision = (true_positive) / (true_positive + false_positive)
    recall = (true_positive) / (true_positive + false_negative)
    f1_measure = (2*true_positive)/((2*true_positive) + false_positive + false_negative)
    return accuracy, precision, recall, f1_measure

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
    accuracy, precision, recall,\
    f1_measure = calculate_accuracy(class_list, test_data_idx)
    #print("Accuracy: ",accuracy, "Precision: ", precision, "Recall: ", recall,
    #"F1-measure: ", f1_measure)
    print("Fold: ",i+1)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_measure_list.append(f1_measure)
accuracy = np.sum(accuracy_list)/len(accuracy_list)
precision = np.sum(precision_list)/len(precision_list)
recall = np.sum(recall_list)/len(recall_list)
f1_measure = np.sum(f1_measure_list)/len(f1_measure_list)
print("Accuracy: ",accuracy, "Precision: ", precision, "Recall: ", recall,
"F1-measure: ", f1_measure)