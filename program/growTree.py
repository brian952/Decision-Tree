import json
import operator

import numpy as np
from math import log
import disp
import evaluate
import pruneTree


def calculate_entropy(data_set):
    count = len(data_set)
    label_counts = {}
    for data in data_set:
        # The last column of each set of data is the category
        current_label = data[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
            # Count how many classes there are and the number of each class
        label_counts[current_label] += 1
    entropy = 0
    for key in label_counts:
        # Calculate the entropy of a single class
        prob = float(label_counts[key]) / count
        # Accumulate the entropy of each class
        entropy -= prob * log(prob, 2)
    return entropy


# Data classified by a feature
def split_data_set(data_set, axis, value):
    result_data_set = []
    for data in data_set:
        if data[axis] == value:
            reduced_data = np.concatenate((data[:axis], data[axis + 1:]))
            result_data_set.append(reduced_data)
    return result_data_set


# Choose the best classification feature
def choose_best_feature_to_split(data_set):
    num_features = len(data_set[0]) - 1
    # Raw entropy
    base_entropy = calculate_entropy(data_set)
    best_info_gain = 0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]
        unique_values = set(feat_list)
        new_entropy = 0
        for value in unique_values:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            # Entropy after feature classification
            new_entropy += prob * calculate_entropy(sub_data_set)

        # The difference between the original entropy
        # and the entropy after feature classification
        info_gain = base_entropy - new_entropy
        # If divided by a feature, the entropy value decreases the most,
        # then this feature is the best classification feature
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


# Sort by number of categories after classification
def majority_count(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return int(sorted_class_count[0][0])


def create_tree(data_set, labels):
    # Category: High-2 or Low-1
    class_list = [data[-1] for data in data_set]
    # When dividing according to a certain category,
    # the labels of each group of data are all the same, no need to continue
    if class_list.count(class_list[0]) == len(class_list):
        return int(class_list[0])
    # The tree building process reached its maximum depth.
    # That is, all the categories used to determine the selection criteria are exhausted
    if len(data_set[0]) == 1:
        return majority_count(class_list)

    # Choose the best feature
    best_feature = choose_best_feature_to_split(data_set)
    best_feature_label = labels[best_feature]
    # Classification results are saved as a dictionary
    decision_tree = [best_feature_label, {}]
    del (labels[best_feature])
    feature_values = [data[best_feature] for data in data_set]
    unique_values = set(feature_values)
    for value in unique_values:
        sub_labels = labels[:]
        decision_tree[1][int(value)] = create_tree(split_data_set(data_set, best_feature, value), sub_labels)

    return decision_tree


def main():
    with open("../data/dataDesc.txt", "r", encoding="utf-8") as f:
        m = json.load(f)

    feature_labels = []
    for i in m:
        feature_labels.append(i[0])
    del (feature_labels[0])

    decision_tree_filename_ = "treeFileFull.txt"
    # Transpose the data, then move the first column to the last column
    train_data_ = np.loadtxt("../data/train.txt", dtype=int).transpose()
    train_data = np.zeros(train_data_.shape)
    train_data[:, 0:len(train_data_[0]) - 1] = train_data_[:, 1:len(train_data_[0])]
    train_data[:, len(train_data_[0]) - 1] = train_data_[:, 0]

    with open(f"../data/{decision_tree_filename_}", "w", encoding="utf=8") as f:
        json.dump(create_tree(train_data, feature_labels), f)

    return decision_tree_filename_


if __name__ == '__main__':
    # write the full tree and pruned tree.
    print("Write full tree              : ", main())
    print("Write pruned tree            : ", pruneTree.main("treeFileFull.txt"))
    # test with the full tree and pruned tree.
    print("The accuracy with full tree  : ", evaluate.main("treeFileFull.txt"))
    print("The accuracy with pruned tree: ", evaluate.main("treeFilePruned.txt"))
    # draw the full tree and pruned tree.
    disp.showIt("treeFileFull.txt")
    print("Draw full tree               : ", "treePicFull.txt")
    disp.showIt("treeFilePruned.txt")
    print("Draw pruned tree             : ", "treePicFull.txt")
