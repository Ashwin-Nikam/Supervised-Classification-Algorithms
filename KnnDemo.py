import sys
import numpy as np

"""
Return [accuracy, precision, recall, f-1 measure] as a list for test tuples identified by their indices
"""


def get_metrics(ground_truth, our_algo):
    metrics = [0.0, 0.0, 0.0, 0.0]
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    last_column = len(ground_truth[0]) - 1
    for i in range(len(ground_truth)):
        if ground_truth[i][last_column] == our_algo[i][last_column] == 1:
            tp += 1
        elif ground_truth[i][last_column] == 0 and our_algo[i][last_column] == 1:
            fp += 1
        elif ground_truth[i][last_column] == 1 and our_algo[i][last_column] == 0:
            fn += 1
        elif ground_truth[i][last_column] == 0 and our_algo[i][last_column] == 0:
            tn += 1

    metrics[0] = (tp + tn) / (tp + tn + fp + fn)  # accuracy
    if tp+fp != 0:
        metrics[1] = tp / (tp + fp)  # precision
    if tp+fn != 0:
        metrics[2] = tp / (tp + fn)  # recall
    if ((2 * tp) + fp + fn) != 0:
      metrics[3] = (2 * tp) / ((2 * tp) + fp + fn)  # f-1 measure
    return metrics


""" Calculates euclidean distance between a and b. If an attribute is nominal, then exact match between that attribute of 
both a and b means adding 1, else adding 0. a and b are vectors of Min-max normalized[0,1] values. Might contain nominal attributes"""


def euclidean_dist(a, b):
    dist = 0.0
    for i in range(len(a) - 1):  # last column is class label, ignore
        if isinstance(a[i], str):
            if a[i] != b[i]:
                dist += 1
        else:
            # print(' + abs(' + a[i] + '-' + b[i] + ')^2')
            dist += pow(a[i] - b[i], 2)
    dist = pow(dist, 0.5)
    # print(dist)
    return dist


def z_score_formula(v, mean, std_dev):
    z_score = (v - mean)/std_dev
    return z_score


# Calculate the z-scores for input data
def z_score_normalize(a):
    normalized = []
    # normalize contents..
    n_features = len(a[0]) - 1  # ignore last column because it is the class label
    means = []
    std_devs = []

    nominal_indices = set()
    for col in range(n_features):
        # perform quick test to see if a[0][col] is a nominal value. if so, skip processing
        colvalues = []
        stat, num = is_number(a[0][col])
        if not stat:
            nominal_indices.add(col)
            continue
        for row in range(len(a)):
            colvalues.append(a[row][col])

        means.append(np.mean(colvalues))
        std_devs.append(np.std(colvalues))

    for row in a:
        new_tuple = ()
        for col in range(n_features):
            if col in nominal_indices:
                new_tuple += (row[col],)
            else:
                new_tuple += (z_score_formula(row[col], means[col], std_devs[col]),)
        # Don't forget to reattach original clas label
        new_tuple += (row[n_features],)
        normalized.append(new_tuple)
    return means, std_devs, normalized


# Calculate the z-scores of the test_input using means and std_devs obtained prior (training data)
def z_score_map(test_input, means, std_devs):
    normalized = []
    n_features = len(test_input[0]) - 1  # ignore last column because it is the class label
    for i in range(len(test_input)):
        new_tuple = ()
        for j in range(n_features):
            stat, num = is_number(test_input[i][j])
            if not stat:
                new_tuple += (test_input[i][j],)
            else:
                new_tuple += (z_score_formula(test_input[i][j], means[j], std_devs[j]),)
        # Don't forget to reattach original clas label
        new_tuple += (test_input[i][n_features],)
        normalized.append(new_tuple)
    return normalized


# Used to differentiate between a nominal attribute and an interval attribute(i.e. a number)
def is_number(n):
    no = 0.0
    try:
        no = float(n)
    except:
        return False, None
    return True, no


# Calculate a mxn matrix. m=len(test data), n = len(train_data)
def calc_dist_matrix(train_data, test_data):
    distances = [[0 for k in (range(len(train_data)))] for i in range(len(test_data))]

    for i in range(len(test_data)):
        for j in range(len(train_data)):
            distances[i][j] = euclidean_dist(test_data[i], train_data[j])

    return distances


# Return (indices of) k nearest neighbors of origin from the entire training data (i.e. those that have been labelled
# correctly(either by our algorithm or by the train data itself)

def get_nearest_neighbors(distances, origin, k):
    neigh = []
    while k > 0:
        minimum = sys.float_info.max
        min_index = -1
        for d in range(len(distances[origin])):
            if d not in neigh:  # origin<->d: d must not be taken already
                if distances[origin][d] < minimum:
                    minimum = distances[origin][d]
                    min_index = d
        neigh.append(min_index)
        k -= 1
    return neigh


"""
Main algorithm. The two parameters(Type: Set) specify which indices to use for training/testing 
"""


def k_nn(normalized_train_input, normalized_test_input, test_input, k):
    last_column = len(normalized_train_input[0]) - 1  # ignore last column because it is the class label
    # Start K-nn

    # Initially labeled data is basically indices of train data. Gradually other points(from test data) get labeled
    # precalculate distances
    distances = calc_dist_matrix(normalized_train_input, normalized_test_input)
    for idx in range(len(normalized_test_input)):
        votes = {0: 0, 1: 0}  # 0 votes for both outcomes (0 and 1)
        neighbors = get_nearest_neighbors(distances, idx, k) ##.union(newly_trained_idx) first param
        # print('\nNeighbors:',neighbors)
        for n in neighbors:
            votes[normalized_train_input[n][last_column]] += 1  # py auto converts votes[0.0]-> votes[0]  and votes[1.0]->votes[1]
        # print('Votes:', votes)
        # assign popular vote
        knn_label = 0
        if votes[1] > votes[0]:
            knn_label = 1
        # print('Item ' + str(pt) + '-> Original Label:' + str((int)(normalized_train_input[pt][last_column])) + ", k-NN Label:" + str(knn_label))

        # update the actual tuple too
        modified_tuple = ()
        for item in normalized_test_input[idx]:
            if item == normalized_test_input[idx][last_column]:
                modified_tuple += (knn_label,)
            else:
                modified_tuple += (item,)
        normalized_test_input.pop(idx)
        normalized_test_input.insert(idx, modified_tuple)

    return get_metrics(test_input, normalized_test_input)


if __name__ == '__main__':
    # Process train data

    f = open('project3_dataset3_train.txt', newline='\n')
    train_input = []
    for line in f:
        line = line.split('\t')
        entry = ()
        entry_nt = []
        for word in line:
            if word == line[len(line) - 1]:
                word = word.rstrip("\n")  # remove trailing \n
            status, no = is_number(word)
            if status:
                entry += (no,)
            else:
                entry += (word,)  # nominal attribute. Store unconverted
        # print(entry)
        train_input.append(entry)

    means, std_devs, normalized_train_input = z_score_normalize(train_input)

    # Process test data

    f2 = open('project3_dataset3_test.txt', newline='\n')

    test_input = []
    for line in f2:
        line = line.split('\t')
        entry = ()
        for word in line:
            if word == line[len(line) - 1]:
                word = word.rstrip("\n")  # remove trailing \n
            status, no = is_number(word)
            if status:
                entry += (no,)
            else:
                entry += (word,)  # nominal attribute. Store unconverted
        # print(entry)
        test_input.append(entry)

    normalized_test_input = z_score_map(test_input, means, std_devs)

    for i in range(15):
        k = i
        metrics = k_nn(normalized_train_input, normalized_test_input, test_input, k)
        print("k = ", k)
        print('Accuracy:', metrics[0])
        print('Precision:', metrics[1])
        print('Recall:', metrics[2])
        print('F-1 measure:', metrics[3])
        print()