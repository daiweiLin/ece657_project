
import numpy as np
import pandas as pd

import Tree


class DecisionTree:
    def __init__(self, data):
        self.data = data
        self.tree = Tree()


    def find_split_val(self,data):
        # Find best split value
        class_values = list(set(row[-1] for row in data))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(data[0]) - 1):
            for row in data:
                groups = self.split(index, row[index], data)
                gini = self.gini_index(groups, class_values)
                print('X%d < %.3f Gini=%.3f' % ((index + 1), row[index], gini))
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def split(self, val, feature, data):
        # split data according to value

        # data with feature value smaller than val goes to left ( feature < val )
        # the rest goes to right

        left = data.loc[data[feature] < val]
        right = data.loc[data[feature] >= val]

        return [left, right]

    def gini_index(self, groups, classes):
        # calculte Gini index of a split
        # sum up gini index i(s,t) of both children trees, ex. i_left + i_right

        n_instances = 0

        for gr in groups:
            n_instances += gr.shape[0]

        # sum weighted Gini index for each group
        gini = 0.0
        for gr in groups:
            size = gr.shape[0]
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0

            # score the group based on the score for each class
            for class_val in classes:
                p = np.sum(gr['class'] == class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances) # *0.5
        return gini

    def train(self):
        # grow a tree
        return None

    def predict(self, data):
        pred = np.zeros((data.shape[0], 1))
        return pred

    def prune(self, tree):
        # prune here
        pruned_tree = tree
        return pruned_tree


if __name__ == '__main__':
    data = pd.read_csv("")
    dt = DecisionTree(data)
    dt.train()
    dt.predict()