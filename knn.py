import math
import sys

"""
Return [accuracy, precision, recall, f-1 measure] as a list for test tuples identified by their indices
"""


def get_metrics(ground_truth, our_algo, indices):
    metrics = [0.0, 0.0, 0.0, 0.0]
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    last_column = len(ground_truth[0]) - 1
    for i in indices:
        if ground_truth[i][last_column] == our_algo[i][last_column] == 1:
            tp += 1
        elif ground_truth[i][last_column] == 0 and our_algo[i][last_column] == 1:
            fp += 1
        elif ground_truth[i][last_column] == 1 and our_algo[i][last_column] == 0:
            fn += 1
        elif ground_truth[i][last_column] == 0 and our_algo[i][last_column] == 0:
            tn += 1

    metrics[0] = (tp + tn) / (tp + tn + fp + fn)  # accuracy
    metrics[1] = tp / (tp + fp)  # precision
    metrics[2] = tp / (tp + fn)  # recall
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
            dist += pow(abs(a[i] - b[i]), 2)
    dist = pow(dist, 0.5)
    # print(dist)
    return dist


# Min max normalization ignoring nominal values as well as the last column (which is the class label)

def min_max_normalize(a):
    normalized = []
    # normalize contents..
    n_features = len(a[0]) - 1  # ignore last column because it is the class label
    mins = []
    maxes = []

    nominal_indices = set()
    for col in range(n_features):
        mins.append(sys.float_info.max)
        maxes.append(sys.float_info.min)

        # perform quick test to see if a[0][col] is a nominal value. if so, skip processing
        stat, num = is_number(a[0][col])
        if not stat:
            nominal_indices.add(col)
            continue

        for row in a:
            if row[col] < mins[col]:
                mins[col] = row[col]
            if row[col] > maxes[col]:
                maxes[col] = row[col]

    # cols_to_normalize = set(range(n_features)).difference(nominal_indices)
    # print('Will normalize columns: ',cols_to_normalize)
    for row in a:
        new_tuple = ()
        for col in range(n_features):
            if col in nominal_indices:
                new_tuple += (row[col],)
            else:
                new_tuple += (min_max_formula(row[col], mins[col], maxes[col]),)
        # Don't forget to reattach original clas label
        new_tuple += (row[n_features],)
        normalized.append(new_tuple)
    return normalized


def min_max_formula(v, min, max):
    return (v - min) / (max - min)


# Used to differentiate between a nominal attribute and an interval attribute(i.e. a number)
def is_number(n):
    no = 0.0
    try:
        no = float(n)
    except:
        return False, None
    return True, no


def calc_dist_matrix(input_data):
    distances = []
    for i in range(len(input_data)):  # initialization with zeros(can't allocate space fo 2d array directly python)
        temp = []
        for j in range(len(input_data)):
            temp.append(0)
        distances.append(temp)

    for i in range(len(input_data)):
        for j in range(i, len(input_data)):
            distances[i][j] = distances[j][i] = euclidean_dist(input_data[i], input_data[j])

    return distances


# Return (indices of) k nearest neighbors of origin from the set labeled_data (i.e. those that have been labelled
# correctly(either by our algorithm or by the train data itself)

def get_nearest_neighbors(labeled_data, distances, origin, k):
    neigh = []
    neigh.append(origin)  # append origin to avoid calculating self distances
    while k > 0:
        minimum = sys.float_info.max
        min_index = -1
        for d in range(len(distances[origin])):
            if d not in neigh and d in labeled_data:  # origin<->d: d must not be taken already AND d should be a labeled point.
                if distances[origin][d] < minimum:
                    minimum = distances[origin][d]
                    min_index = d
        neigh.append(min_index)
        k -= 1
    neigh.remove(origin)  # restore neigh
    return neigh


"""
Main algorithm. The two parameters(Type: Set) specify which indices to use for training/testing 
"""


def k_nn(train_data_idx, test_data_idx, input_data, normalized_input):
    # euclidean_dist(input_data[0], input_data[1])
    # euclidean_dist(normalized_input[2], normalized_input[1])
    print()
    # print(normalized_input)
    last_column = len(normalized_input[0]) - 1  # ignore last column because it is the class label
    # Start K-nn

    k = 4
    # Initially labeled data is basically indices of train data. Gradually other points(from test data) get labeled
    # precalculate distances
    distances = calc_dist_matrix(normalized_input)
    newly_trained_idx = set()
    for pt in test_data_idx:
        votes = {0: 0, 1: 0}  # 0 votes for both outcomes (0 and 1)
        neighbors = get_nearest_neighbors(train_data_idx.union(newly_trained_idx), distances, pt, k)
        # print('\nNeighbors:',neighbors)
        for n in neighbors:
            votes[normalized_input[n][
                last_column]] += 1  # py auto converts votes[0.0]-> votes[0]  and votes[1.0]->votes[1]
        # print('Votes:', votes)
        # assign popular vote
        knn_label = 0
        if votes[1] > votes[0]:
            knn_label = 1
        # print('Item ' + str(pt) + '-> Original Label:' + str((int)(input_data[pt][last_column])) + ", k-NN Label:" + str(knn_label))

        # Mark pt as labeled
        newly_trained_idx.add(pt)

        # update the actual tuple too! Otherwise future votes will be incorrect
        modified_tuple = ()
        for item in normalized_input[pt]:
            if item == normalized_input[pt][last_column]:
                modified_tuple += (knn_label,)
            else:
                modified_tuple += (item,)
        normalized_input.pop(pt)
        normalized_input.insert(pt, modified_tuple)

    # Uncomment for demoing knn without k-fold cross validation:
    # metrics = get_metrics(input_data, normalized_input)
    # print('Accuracy:', metrics[0])
    # print('Precision:', metrics[1])
    # print('Recall:', metrics[2])
    # print('F-1 measure:', metrics[3])
    return get_metrics(input_data, normalized_input, test_data_idx)


if __name__ == '__main__':
    f = open('project3_dataset2.txt', newline='\n')
    input_data = []
    for line in f:
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
        input_data.append(entry)
    normalized_input = min_max_normalize(input_data)

    # 10-fold cross validation:

    folds = 10
    part_len = int(len(normalized_input) / folds)
    metrics_avg = [0.0,0.0,0.0,0.0]
    train_data_idx = set()

    for i in range(folds):
        if i != folds - 1:
            start = (i * part_len)
            end = start + part_len
            test_data_idx = set(range(start, end))

        else:
            test_data_idx = set(range(i * part_len, len(normalized_input)))

        train_data_idx = set(range(len(normalized_input))).difference(test_data_idx)
        # print('\nTest',test_data_idx,':'+ str(len(test_data_idx)))
        # print('Train', train_data_idx, ':' + str(len(train_data_idx)))
        metrics = k_nn(train_data_idx, test_data_idx, input_data, normalized_input)
        metrics_avg[0] += metrics[0]
        metrics_avg[1] += metrics[1]
        metrics_avg[2] += metrics[2]
        metrics_avg[3] += metrics[3]

        print('Iteration ', (i + 1))
        print('Accuracy:', metrics[0])
        print('Precision:', metrics[1])
        print('Recall:', metrics[2])
        print('F-1 measure:', metrics[3])

    metrics_avg[0] /= folds
    metrics_avg[1] /= folds
    metrics_avg[2] /= folds
    metrics_avg[3] /= folds

    print('\nAverage Accuracy:', metrics_avg[0])
    print('Average Precision:', metrics_avg[1])
    print('Average Recall:', metrics_avg[2])
    print('Average F-1 measure:', metrics_avg[3])

    # TODO Verify results.