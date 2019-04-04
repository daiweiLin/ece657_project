"""
Author: Daiwei Lin
Date: 2019/04/01

ECE657A project

"""

import numpy as np
import pandas as pd
import math
from sklearn.base import BaseEstimator, ClassifierMixin


class DecisionTreeC45Node(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth, cat_features=None):
        # property
        self.max_depth = max_depth

        # constant
        self.cat_features = cat_features
        # variables
        self.children = list()
        self.isleaf = False

        self.split_val = None
        self.split_feature = None
        self.is_cat_feature = False
        self.classification = None

        self.depth = 0

    def is_pure(self, y):
        # check whether the branch is pure() having same class )

        # compare all class label with the class of first row.
        #         print(y)
        for i in y:
            if i != y[0]:
                return False
        return True

    def find_split_val(self, X, y, potential_features):
        # Find best split value

        class_values = list(set(y))
        b_feature, b_value, b_score, b_groups = None, None, None, None

        entropy_parent = self.entropy([y], class_values)
        for feature in potential_features:
            if feature in self.cat_features:
                groups = self.split(feature, None, X, y)
                groups_y = list()
                split_values = list()
                for v, gr in groups.items():
                    groups_y.append(gr['y'])
                    split_values.append(v)
                entropy_children = self.entropy(groups_y, class_values)
                split_info = self.splitting_information(groups_y)
                if split_info == 0:
                    gain_ratio = 0
                else:
                    gain_ratio = (entropy_parent - entropy_children)
                #                 print('X{:d} ={} E={:.3f},IG={:.3f},SI={:.3f},GR={:.3f}'.format((feature + 1),
                #                                                                                 split_values,
                #                                                                                 entropy_children,
                #                                                                                 (entropy_parent - entropy_children),
                #                                                                                 split_info, gain_ratio))
                if b_score is None or gain_ratio > b_score:
                    b_index, b_value, b_score, b_groups, is_cat_feature = feature, split_values, gain_ratio, groups, True
            else:
                for row in X:
                    groups = self.split(feature, row[feature], X, y)
                    groups_y = list()
                    for _, gr in groups.items():
                        groups_y.append(gr['y'])
                    entropy_children = self.entropy(groups_y, class_values)
                    split_info = self.splitting_information(groups_y)
                    if split_info == 0:
                        gain_ratio = 0
                    else:
                        gain_ratio = (entropy_parent - entropy_children)
                    # print('X{:d} ={} E={:.3f},IG={:.3f},SI={:.3f},GR={:.3f}'.format((feature + 1), row[feature], entropy_children, (entropy_parent - entropy_children), split_info, gain_ratio))
                    if b_score is None or gain_ratio > b_score:
                        b_index, b_value, b_score, b_groups, is_cat_feature = feature, row[
                            feature], gain_ratio, groups, False
        return {'index': b_index, 'value': b_value, 'groups': b_groups, 'gain_ratio': b_score,
                'is_cat_feature': is_cat_feature}

    def split(self, feature, val, X, y):
        # split data according to split criteria
        groups = dict()
        if feature in self.cat_features:
            # categorical feature ( multiple branches )
            # create branches for each category
            for idx, row in enumerate(X):
                if row[feature] in groups.keys():
                    groups[row[feature]]['X'].append(row)
                    groups[row[feature]]['y'].append(y[idx])
                else:
                    groups[row[feature]] = {'X': [row], 'y': [y[idx]]}
        else:
            # numerical feature (binary branches)
            # data with feature value smaller than val goes to left ( feature < val )
            # the rest goes to right
            groups['left'] = {'X': list(), 'y': list()}
            groups['right'] = {'X': list(), 'y': list()}
            for idx, row in enumerate(X):
                if row[feature] < val:
                    groups['left']['X'].append(row)
                    groups['left']['y'].append(y[idx])
                else:
                    groups['right']['X'].append(row)
                    groups['right']['y'].append(y[idx])

        return groups

    def entropy(self, groups, classes):
        # calculte entropy of a split
        n_instances = 0
        for gr in groups:
            n_instances += len(gr)

        entropy = 0.0
        for gr in groups:
            size = len(gr)
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0

            # score the group based on the score for each class
            for class_val in classes:
                p = 0.0
                for v in gr:
                    if v == class_val:
                        p += 1
                p = p / size
                if p > 0:
                    score += p * math.log2(p)
            # weight the group score by its relative size
            entropy += (score) * (size / n_instances)  # *0.5
        entropy = -entropy
        return entropy

    def splitting_information(self, groups):
        n_instances = 0
        for gr in groups:
            n_instances += len(gr)
        #             print(len(gr))

        splitting_info = 0.0
        for gr in groups:
            t = len(gr) / n_instances
            if t > 0.0:
                splitting_info += t * np.log2(t)
        splitting_info = -splitting_info
        return splitting_info

    def grow(self, X, y, depth, potential_features):
        #         print("\ndepth={}, p_features={}".format(depth,potential_features))
        if self.is_pure(y) or self.max_depth <= depth or len(potential_features) == 0:
            #             print("terminate at depth={}".format(depth))
            self.terminate(y, depth)
            return
        else:

            best_split = self.find_split_val(X, y, potential_features)
            self.split_val = best_split['value']
            self.split_feature = best_split['index']
            self.is_cat_feature = best_split['is_cat_feature']
            # print("split on feature:{} GR={}".format(self.split_feature, best_split['gain_ratio']))

            if self.is_cat_feature:
                potential_features.remove(self.split_feature)
            for _, gr in best_split['groups'].items():
                node = DecisionTreeC45Node(self.max_depth, self.cat_features)
                node.grow(gr['X'], gr['y'], depth + 1, potential_features)
                self.children.append(node)
                #             print("{}X{} < {} gini={}".format(self.depth*' ',self.split_feature+1, self.split_val, best_split['gini']))
            #             print("left - Class 0:{},class 1:{}".format(np.sum(left.iloc[:,-1]==0),np.sum(left.iloc[:,-1]==1)))
            #             print("right - Class 0:{},class 1:{}".format(np.sum(right.iloc[:,-1]==0),np.sum(right.iloc[:,-1]==1)))

        self.depth = depth
        return

    def terminate(self, y, depth):
        # define leaf node
        # most frequent class in the data as class label of this node
        self.classification = max(set(y), key=y.count)
        self.isleaf = True
        self.depth = depth

    def fit(self, X, y):
        # grow a tree
        n = len(X.columns)

        # convert cat_feature to its index in data X's columns
        if self.cat_features is None:
            self.cat_features = []
        else:
            i = 0
            cat_features_idx = []
            for col in X.columns:
                if col in self.cat_features:
                    cat_features_idx.append(i)
                i += 1
            self.cat_features = cat_features_idx

        X = X.values.tolist()
        y = y.values.tolist()
        potential_features = np.linspace(0, n - 1, n, dtype=np.int32).tolist()
        self.grow(X, y, 1, potential_features)
        return self

    def print_tree(self):
        if not self.isleaf:
            if self.is_cat_feature:
                for i in range(len(self.split_val)):
                    print("{}X{} = {} ".format(self.depth * ' ', self.split_feature, self.split_val[i]))
                    self.children[i].print_tree()
            else:
                print("{}X{} < {} ".format(self.depth * ' ', self.split_feature, self.split_val))
                self.children[0].print_tree()
                self.children[1].print_tree()
        else:
            print("{}[{}]".format(self.depth * ' ', self.classification))

    def predict_iterate(self, row):

        if self.isleaf:
            # is leaf node
            return self.classification
        else:
            # not leaf node
            if self.is_cat_feature:
                # predict categorical feature
                for i in range(len(self.split_val)):
                    if row[self.split_feature] == self.split_val[i]:
                        return self.children[i].predict_iterate(row)
                return None
            else:
                # predict numerical feature
                if row[self.split_feature] < self.split_val:
                    return self.children[0].predict_iterate(row)
                else:
                    return self.children[1].predict_iterate(row)

    def predict(self, X):
        num_rows = X.shape[0]
        prediction = []
        #         prediction = np.zeros((num_rows,1))
        for idx, row in X.iterrows():
            prediction.append(self.predict_iterate(row))

        return np.array(prediction)

    def prune(self, tree):
        # prune here
        pruned_tree = tree
        return pruned_tree
