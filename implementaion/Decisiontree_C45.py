"""
Author: Daiwei Lin
Date: 2019/04/02

ECE657A project

"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class DecisionTreeC45Node(BaseEstimator, ClassifierMixin):
    """ CART classifier implementation in sklearn standard

    The implementation is adapted to sklearn standard, in
    order to make this classifier usable in GridsearchCV()

    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.

    """

    def __init__(self, max_depth, cat_features=None):
        # property
        self.max_depth = max_depth

        # constant
        self.cat_features = cat_features
        # variables
        self.left = None
        self.right = None
        self.isleaf = False

        self.split_val = None
        self.split_feature = None
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

    def find_split_val(self, X, y):
        # Find best split value
        class_values = list(set(y))
        b_feature, b_value, b_score, b_groups = None, None, None, None
        for feature in range(len(X[0])):
            for row in X:
                groups = self.split(feature, row[feature], X, y)
                gini = self.gini_index([groups[0]['y'], groups[1]['y']], class_values)
                #                 print('X%d < %.3f Gini=%.3f' % ((feature + 1), row[feature], gini))
                if b_score is None or gini < b_score:
                    b_index, b_value, b_score, b_groups = feature, row[feature], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups, 'gini': b_score}

    def split(self, feature, val, X, y):
        # split data according to split criteria
        left_X, left_y, right_X, right_y = list(), list(), list(), list()
        if feature in self.cat_features:
            # categorical feature
            # data with feature value equal to val goes to left ( feature == val )
            # the rest goes to right
            for idx, row in enumerate(X):
                if row[feature] == val:
                    left_X.append(row)
                    left_y.append(y[idx])
                else:
                    right_X.append(row)
                    right_y.append(y[idx])
        #             idx = X[feature] == val
        #             left_X = X.loc[idx]
        #             left_y = y.loc[idx]
        #             idx = ~idx
        #             right_X = X.loc[idx]
        #             right_y = y.loc[idx]
        else:
            # numerical feature
            # data with feature value smaller than val goes to left ( feature < val )
            # the rest goes to right
            for idx, row in enumerate(X):
                if row[feature] < val:
                    left_X.append(row)
                    left_y.append(y[idx])
                else:
                    right_X.append(row)
                    right_y.append(y[idx])

        return [{'X': left_X, 'y': left_y}, {'X': right_X, 'y': right_y}]

    def entropy(self, groups, classes):




    def gini_index(self, groups, classes):
        # calculte Gini index of a split
        # sum up gini index i(s,t) of both children trees, ex. i_left + i_right

        n_instances = 0

        for gr in groups:
            n_instances += len(gr)

        # sum weighted Gini index for each group
        gini = 0.0
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
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)  # *0.5
        return gini

    def grow(self, X, y, depth):

        if self.is_pure(y) or self.max_depth <= depth:
            #             print("terminate at depth={}".format(depth))
            self.terminate(y, depth)
            return
        else:

            best_split = self.find_split_val(X, y)
            self.split_val = best_split['value']
            self.split_feature = best_split['index']
            [left, right] = best_split['groups']

            #             print("{}X{} < {} gini={}".format(self.depth*' ',self.split_feature+1, self.split_val, best_split['gini']))
            #             print("left - Class 0:{},class 1:{}".format(np.sum(left.iloc[:,-1]==0),np.sum(left.iloc[:,-1]==1)))
            #             print("right - Class 0:{},class 1:{}".format(np.sum(right.iloc[:,-1]==0),np.sum(right.iloc[:,-1]==1)))

            self.left = DecisionTreeC45Node(self.max_depth, self.cat_features)
            self.right = DecisionTreeC45Node(self.max_depth, self.cat_features)
            self.left.grow(left['X'], left['y'], depth + 1)
            self.right.grow(right['X'], right['y'], depth + 1)

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
        X = X.values.tolist()
        y = y.values.tolist()
        self.grow(X, y, 1)
        return self

    def print_tree(self):
        if not self.isleaf:
            if self.split_feature in self.cat_features:
                print("{}X{} = {} ".format(self.depth * ' ', self.split_feature, self.split_val))
            else:
                print("{}X{} < {} ".format(self.depth * ' ', self.split_feature, self.split_val))
            self.left.print_tree()
            self.right.print_tree()
        else:
            print("{}[{}]".format(self.depth * ' ', self.classification))

    def predict_iterate(self, row):

        if self.isleaf:
            # is leaf node
            return self.classification
        else:
            # not leaf node
            if self.split_feature in self.cat_features:
                # predict categorical feature
                if row[self.split_feature] == self.split_val:
                    return self.left.predict_iterate(row)
                else:
                    return self.right.predict_iterate(row)
            else:
                # predict numerical feature
                if row[self.split_feature] < self.split_val:
                    return self.left.predict_iterate(row)
                else:
                    return self.right.predict_iterate(row)

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


if __name__ == '__main__':
    data = pd.DataFrame([[2.771244718,1.784783929, 'a',0],
                         [1.728571309,1.169761413, 'a',0],
                         [3.678319846,2.81281357, 'a',0],
                         [3.961043357,2.61995032, 'a',0],
                         [2.999208922,2.209014212, 'a',0],
                         [7.497545867,3.162953546, 'a',1],
                         [9.00220326,3.339047188, 'a',1],
                         [7.444542326,0.476683375, 'a',1],
                         [10.12493903,3.234550982, 'a',1],
                         [6.642287351,3.319983761, 'a',1]])

    dt = DecisionTreeC45Node(max_depth=5, cat_features=[2])
    # %prun dt.train(X=X_train,y=y_train, cat_features=str_cols, max_depth=5)
    dt.fit(X=data.iloc[:,:-1], y=data.iloc[:,-1])
    pred = dt.predict(data.iloc[:, :-1])
    print(pred)