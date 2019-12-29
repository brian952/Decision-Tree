import json

import numpy as np


def predict(tree, data_to_test, li):
    if type(tree) != list:
        return tree
    label = tree.copy()
    while type(label) == list:
        for i in range(len(li)):
            if label[0] == li[i][0]:
                break
        label = label[1][str(data_to_test[i - 1])]
    return label


def main(fname):
    test_data = np.loadtxt("../data/test.txt", dtype=int)
    with open('../data/' + fname) as f:
        tree = json.load(f)
    with open('../data/dataDesc.txt') as f:
        data_desc = json.load(f)

    accuracy = 0
    for i in range(np.size(test_data, axis=1)):
        if predict(tree, test_data[1:, i], data_desc) == test_data[0, i]:
            accuracy += 1
    return accuracy / np.size(test_data, axis=1)


if __name__ == '__main__':
    print("The accuracy with full tree  : ", main("treeFileFull.txt"))
    print("The accuracy with pruned tree: ", main("treeFilePruned.txt"))
