import numpy as np
from typing import List
from classifier import Classifier


class DecisionTree(Classifier):
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features: List[List[float]], labels: List[int]):
        # init.
        assert (len(features) > 0)
        self.feautre_dim = len(features[0])
        num_cls = np.max(labels) + 1

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features: List[List[float]]) -> List[int]:
        y_pred = []
        for feature in features:
            y_pred.append(self.root_node.predict(feature))
        return y_pred

    def print_tree(self, node=None, name='node 0', indent=''):
        if node is None:
            node = self.root_node
        print(name + '{')

        string = ''
        for idx_cls in range(node.num_cls):
            # This has an error!!!
            string += str(node.labels.count(idx_cls)) + ' '
        print(indent + ' num of sample / cls: ' + string)

        if node.splittable:
            print(indent + '  split by dim {:d}'.format(node.dim_split))
            for idx_child, child in enumerate(node.children):
                self.print_tree(node=child, name='  ' + name + '/' + str(idx_child), indent=indent + '  ')
        else:
            print(indent + '  cls', node.cls_max)
        print(indent + '}')


class TreeNode(object):
    def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls

        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label  # majority of current node

        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    def split(self):
        def conditional_entropy(branches: List[List[int]]) -> float:
            '''
            branches: C x B array,
                      C is the number of classes,
                      B is the number of branches
                      it stores the number of
                      corresponding training samples
                      e.g.
                                  ○ ○ ○ ○
                                  ● ● ● ●
                                ┏━━━━┻━━━━┓
                               ○ ○       ○ ○
                               ● ● ● ●

                      branches = [[2,2], [4,0]]
            '''
            entropy_total = 0.0
            branches = np.array(branches)
            C = branches.shape[0]
            B = branches.shape[1]
            for i in range(B):
                branch = branches[:, i]
                sum_no = sum(branch)
                branch = np.multiply(branch, 1.0 / sum_no)
                # print("branch = ", branch)
                entropy_branch = []
                for j in range(len(branch)):
                    if (branch[j] == 0):
                        entropy_branch.append(0.0)
                    else:
                        entropy_branch.append(-branch[j] * np.log(branch[j]))
                # print("branch = ", branch)
                # print("entropy_branch = ", np.sum(entropy_branch))
                entropy_total += np.sum(entropy_branch)
            return entropy_total

        ########################################################
        # TODO: compute the conditional entropy
        ########################################################
        # print ("self.features = ",self.features)
        # print ("self.labels = ", self.labels)
        # print ("self.features[0] = ", self.features[0])

        best_idx_dim = -1
        best_split_idx_dim_featureCount = []
        min_entropy = float('inf')
        N = len(self.features)
        features = np.array(self.features)
        for idx_dim in range(len(self.features[0])):
            featureCount = np.unique(features[:, idx_dim]).tolist()
            labelCount = np.unique(self.labels).tolist()

            totalFeatureNum = len(featureCount)
            maxClassLabel = len(labelCount)

            if (totalFeatureNum == 1):
                continue
            branches = np.zeros((maxClassLabel, totalFeatureNum))
            for i in range(N):
                b = featureCount.index(self.features[i][idx_dim])
                a = labelCount.index(self.labels[i])
                # print("a = ",a,",b = ",b)
                branches[a, b] = branches[a, b] + 1
            # print("branches = ", branches)
            this_split_conditional_entropy = conditional_entropy(branches.tolist())
            # print ("this_split_conditional_entropy = ",this_split_conditional_entropy)
            if (this_split_conditional_entropy < min_entropy):
                min_entropy = this_split_conditional_entropy
                best_idx_dim = idx_dim
                best_split_idx_dim_featureCount = featureCount
        # print("best_split_idx_dim = ", best_idx_dim)

        ############################################################
        # TODO: compare each split using conditional entropy
        #       find the best split
        ############################################################
        # Since best_split_idx_dim is known.
        # print("best_split_idx_dim_featureCount = ", best_split_idx_dim_featureCount)

        if (best_idx_dim != -1):
            self.splittable = True
            for i in range(len(best_split_idx_dim_featureCount)):
                childfeatures = []
                childlabels = []
                label_count = {}
                for j in range(N):
                    if (self.features[j][best_idx_dim] == best_split_idx_dim_featureCount[i]):
                        childfeatures.append(self.features[j])
                        childlabels.append(self.labels[j])
                        label_count[self.labels[j]] = label_count.get(self.labels[j], 0) + 1
                        # if (self.labels[j] not in label_count):
                        #     label_count.get(self.labels[j])
                # print ("childfeatures = ",childfeatures)
                # print("childfeatures.length = ", len(childfeatures))
                # print ("childlabels = ",childlabels)
                # print ("label_count = ",label_count)
                child_num_cls = len(label_count)
                # print("child_num_cls = ", label_count, "child_num_cls = ", child_num_cls)
                # print ("child_num_cls = ",child_num_cls)
                childTree = TreeNode(childfeatures, childlabels, child_num_cls)
                self.children.append(childTree)
                # print ("len(finalfeatureCount) = ",len(finalfeatureCount))
                # print ("len(finallabelCount) = ",len(finallabelCount))
        else:
            best_idx_dim = 0 # no meaning here in best_idx_dim.
            self.splittable = False

        self.dim_split = best_idx_dim
        self.feature_uniq_split = best_split_idx_dim_featureCount

        ############################################################
        # TODO: split the node, add child nodes
        ############################################################
        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()

        return

    def predict(self, feature: List[int]) -> int:
        if self.splittable:
            # print(feature)
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max
