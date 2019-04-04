"""
Author: Daiwei Lin
Date: 2019/04/01

ECE657A project

"""

import numpy as np
import pandas as pd
import math
import itertools
import copy
from scipy.stats import chi2_contingency

from sklearn.base import BaseEstimator, ClassifierMixin


class DecisionTreeCHAIDNode(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth, alpha_merge, alpha_stop, cat_features=None, ):
        # property
        self.max_depth = max_depth

        # constant
        self.cat_features = cat_features
        self.alpha_merge = alpha_merge
        self.alpha_stop = alpha_stop

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

    def find_split_val(self, data, potential_features):
        # Find best split

        b_feature, b_value, b_score, b_groups = None, None, None, None


        for feature in potential_features:
            b_groups_of_cat, adj_p = self.merge(feature, data)
            print("feature:{} adj_p:{}, groups:{}".format(feature,adj_p, b_groups_of_cat))

            if b_score is None or adj_p < b_score:
                if b_groups_of_cat is None:
                    groups = None
                else:
                    groups = self.split(feature, b_groups_of_cat, data)

                b_index, b_value, b_score, b_groups = feature, b_groups_of_cat, adj_p, groups

        return {'index': b_index, 'value': b_value, 'groups': b_groups, 'adj_p': b_score}

    def split(self, feature, groups_of_cat, data):
        # split data according to split criteria

        # create one dic pair for each group in split value (groups_of_cat)
        groups = dict()
        groups_len = len(groups_of_cat)
        for i in range(groups_len):
            groups[i] = list()

        selected_feature = data[feature].tolist()

        for f_idx in range(len(selected_feature)):
            for g_idx in range(groups_len):
                if selected_feature[f_idx] in groups_of_cat[g_idx]:
                    groups[g_idx].append(f_idx)
                    break

        for g_idx, f_idx in groups.items():
            groups[g_idx] = data.iloc[f_idx]

        return groups


        # if feature in self.cat_features:
        # categorical feature ( multiple branches )
        # create branches for each category
        # for idx, row in enumerate(X):
        #     if row[feature] in groups.keys():
        #         groups[row[feature]]['X'].append(row)
        #         groups[row[feature]]['y'].append(y[idx])
        #     else:
        #         groups[row[feature]] = {'X': [row], 'y': [y[idx]]}
        # else:
        #     # numerical feature (binary branches)
        #     # data with feature value smaller than val goes to left ( feature < val )
        #     # the rest goes to right
        #     groups['left'] = {'X': list(), 'y': list()}
        #     groups['right'] = {'X': list(), 'y': list()}
        #     for idx, row in enumerate(X):
        #         if row[feature] < val:
        #             groups['left']['X'].append(row)
        #             groups['left']['y'].append(y[idx])
        #         else:
        #             groups['right']['X'].append(row)
        #             groups['right']['y'].append(y[idx])

        # return groups

    def bonf_adjust(self, p, c, r):
        '''

        :param p: p value
        :param c: number of original categories
        :param r: number of merged categories
        :return: adjusted p value
        '''
        B = 0.0
        for i in range(r):
            B += math.pow(-1,i) * math.pow(r-i, c) / (math.factorial(r) * math.factorial(r-i))
        p * B
#         print("p {}, c {}, r {}, B {}, adj_p {}".format(p,c,r,B, B*p))
        return p * B

    def rm_empty_col(self, c_table):
        # delete columns with all 0 in a contingency table
        # take NUMPY ARRAY as input

        # print(c_table)
        i = 0
        for col in c_table.T:
            if np.sum(col) == 0:
                # print("delete col {}".format(i))
                c_table = np.delete(c_table, i, axis=1)
                i -= 1
            i += 1
        return c_table

    def merge_single_comb(self, c_table, comb):
        # merge contingency table rows according to single combination
        cat_A = comb[0]
        cat_B = comb[1]
        c_table.loc[cat_A,] = c_table.loc[cat_A,] + c_table.loc[cat_B,]
        c_table = c_table.drop(labels=[cat_B], axis=0)
        c_table = c_table.rename(index={cat_A: str(cat_A)+','+str(cat_B)})

        return c_table

    def merge(self, feature, data):
        # find best merge of categories in a given feature
        # Return: b_groups: nested list of merged result
        #         adj_p: adjusted p value of merged result

        contingency_table = pd.crosstab(data[feature], data['class'])
        unique_category = contingency_table.index.tolist()
#         print("contingency table: \n {}".format(contingency_table))
#         print("feature:{}, unique_category:{}".format(feature,unique_category))
        b_groups, adj_p = None, None
        c = len(unique_category)

        if c == 1:
            adj_p = 1
            return b_groups, adj_p

        elif c == 2:
            sub_table = self.rm_empty_col(contingency_table.values)
            chi2, p, _, _ = chi2_contingency(sub_table)
            b_groups = list()
            for cat in unique_category:
                b_groups.append([cat])
            adj_p = p
            return b_groups, adj_p

        else:

            while True:

                # Get the best combination with
                # highest p value for all pairs of categories
                b_comb, b_pvalue = None, None
                for comb in itertools.combinations(unique_category,2):
                    sub_table = contingency_table.loc[list(comb)]
                    sub_table = self.rm_empty_col(sub_table.values)

                    chi2, p, _, _ = chi2_contingency(sub_table)
                    if b_pvalue is None or b_pvalue < p:
                        b_comb, b_pvalue = list(comb), p

                if b_pvalue < self.alpha_merge:
                    # stop if best p is smaller than alpha_merge
                    break
                else:
                    # merge selected combination
                    # print("merge {}".format(b_comb))
                    contingency_table = self.merge_single_comb(contingency_table, b_comb)
                    unique_category = contingency_table.index.tolist()
                    # print("unique category: {}".format(unique_category))
                    # stop merging if only 2 categories left
                    if len(unique_category) <= 2:
                        break

            # Get p value for final merged contigency_table
            chi2, p, _, _ = chi2_contingency(contingency_table)
            # Bonferroni adjustment of p value
            r = len(unique_category)
            adj_p = self.bonf_adjust(p,c,r)
            b_groups = list()
            print(unique_category)
            for gr in unique_category:
                b_groups.append(gr.split(','))

        return b_groups, adj_p

    def grow(self, data, depth, potential_features):
        #         print("\ndepth={}, p_features={}".format(depth,potential_features))
        y = data['class'].tolist()
        if self.is_pure(y) or self.max_depth <= depth or len(potential_features) == 0:
            #             print("terminate at depth={}".format(depth))
            self.terminate(y, depth)
            return
        else:

            best_split = self.find_split_val(data, potential_features)

            if best_split['adj_p'] <= self.alpha_stop:
                self.split_val = best_split['value']
                self.split_feature = best_split['index']
                # self.is_cat_feature = best_split['is_cat_feature']
                print("\nsplit on feature:{} adj_p={}".format(self.split_feature, best_split['adj_p']))



                for idx, gr in best_split['groups'].items():
                    node = DecisionTreeCHAIDNode(self.max_depth,self.alpha_merge,self.alpha_stop)
                    new_potential_features = copy.deepcopy(potential_features)

                    if len(self.split_val[idx]) == 1:
                        print(self.split_val[idx])
                        print(new_potential_features)
                        new_potential_features.remove(self.split_feature)
                    print("---- new node with features:{}, split_val:{}".format(new_potential_features,self.split_val[idx]))
                    node.grow(gr, depth + 1, new_potential_features)
                    self.children.append(node)
            else:
                print("adj_p {} larger than alpha stop".format(best_split['adj_p']))
                self.terminate(y, depth)
                #             print("{}X{} < {} gini={}".format(self.depth*' ',self.split_feature+1, self.split_val, best_split['gini']))


        self.depth = depth
        return

    def terminate(self, y, depth):
        # define leaf node
        # most frequent class in the data as class label of this node
        print("terminate at depth {}".format(depth))
        self.classification = max(set(y), key=y.count)
        self.isleaf = True
        self.depth = depth

    def num_to_cat(self, data, num_cols, k):
        # convert numerical features into categorical
        # using binning (quantile cut into k bins)
        labels = [str(x) for x in range(k)]
        for col in num_cols:
            data[col] = pd.qcut(data[col], k, labels=labels)


    def fit(self, X, y):
        # grow a tree
        # n = len(X.columns)

        # # convert cat_feature to its index in data X's columns
        # if self.cat_features is None:
        #     self.cat_features = []
        # else:
        #     i = 0
        #     cat_features_idx = []
        #     for col in X.columns:
        #         if col in self.cat_features:
        #             cat_features_idx.append(i)
        #         i += 1
        #     self.cat_features = cat_features_idx

#         X = X.values.tolist()
#         y = y.values.tolist()
        potential_features = X.columns.tolist()
        X['class'] = y
        data = X
        # potential_features = np.linspace(0, n - 1, n, dtype=np.int32).tolist()

        self.grow(data, 1, potential_features)
        return self

    def print_tree(self):
        if not self.isleaf:
#             if self.is_cat_feature:
            for i in range(len(self.split_val)):
                print("{}X{} = {} ".format(self.depth * ' ', self.split_feature, self.split_val[i]))
                self.children[i].print_tree()
        else:
            print("{}[{}]".format(self.depth * ' ', self.classification))

    def predict_iterate(self, row):

        if self.isleaf:
            # is leaf node
            return self.classification
        else:
            # not leaf node
            # predict categorical feature
            for i in range(len(self.split_val)):
                if row[self.split_feature] in self.split_val[i]:
                    return self.children[i].predict_iterate(row)
            return None

    def predict(self, X):
        num_rows = X.shape[0]
        prediction = []
        #         prediction = np.zeros((num_rows,1))
        for _, row in X.iterrows():
            prediction.append(self.predict_iterate(row))

        return np.array(prediction)
