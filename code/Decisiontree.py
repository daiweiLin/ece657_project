
import numpy as np
import pandas as pd

import Tree


class DecisionTreeNode:
    def __init__(self, data):
        self.data = data

        self.left = None
        self.right = None
        self.isleaf = False
        self.split_val = None
        self.split_feature = None



    def find_split_val(self):
        # Find best split value
        class_values = list(set(self.data.iloc[:,-1]))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for feature in self.data.columns:
            for _, row in self.data.iterrows():
                groups = self.split(feature, row[feature], self.data)
                gini = self.gini_index(groups, class_values)
                print('X%d < %.3f Gini=%.3f' % ((feature + 1), row[feature], gini))
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = feature, row[feature], gini, groups
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

    def train(self, data):
        # grow a tree
        root = self.get_split(data)
        self.split(root, 1)
        return root

    def predict(self, data):
        pred = np.zeros((data.shape[0], 1))
        return pred

    def prune(self, tree):
        # prune here
        pruned_tree = tree
        return pruned_tree


if __name__ == '__main__':
    data = pd.DataFrame([[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]])

    dt = DecisionTreeNode(data)
    dt.find_split_val()