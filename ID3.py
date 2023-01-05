import numpy as np
# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}

# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label

class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via an Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

        # further variables and functions can be added...


    def calculate_entropy__(self, dataset, labels):
        """
        :param dataset: array of the data instances
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """
        entropy_value = 0.0

        """
        Entropy calculations
        """
        length=len(labels)
        labels=np.array(labels).astype(int)
        p_plus=np.sum(labels)
        p_plus=p_plus/length
        p_minus=1-p_plus
        entropy_value= np.nan_to_num(-1*p_plus*np.log2(p_plus)) - np.nan_to_num(p_minus*np.log2(p_minus))
        return entropy_value

    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated...
        :return: the calculated average entropy value for the given attribute
        """
        average_entropy = 0.0
        """
            Average entropy calculations
        """
        entropy=[]
        length=len(labels)
        full_data=np.column_stack((dataset,labels))
        index = np.where(np.array(self.features) == attribute)[0][0]
        column = full_data[:, index]
        unique,counts= np.unique(column,return_counts=True)
        labels=[]
        for value in unique:
            for rows in full_data:
                if (rows[index]==value):
                    labels.append(int(rows[4]))
            entropy.append(self.calculate_entropy__(dataset,labels))
            labels=[]
        for i in range(len(unique)):
            average_entropy+=(counts[i]/length)*entropy[i]
        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        information_gain = self.calculate_entropy__(dataset, labels) - self.calculate_average_entropy__(dataset, labels, attribute)
        """
            Information gain calculations
        """
        return information_gain

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        intrinsic_info = None
        """
            Intrinsic information calculations for a given attribute
        """
        ratio=0
        length=len(labels)
        full_data=np.column_stack((dataset,labels))
        index = np.where(np.array(self.features) == attribute)[0][0]
        column = full_data[:, index]
        unique,counts= np.unique(column,return_counts=True)
        for i in range(len(unique)):
            ratio=np.nan_to_num(counts[i]/length)
            intrinsic_info+= np.nan_to_num(-ratio*np.log2(ratio))

        return intrinsic_info
    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated...
        :return: the calculated gain ratio score
        """
        """
            Your implementation
        """
        gain_ratio=self.calculate_information_gain__(dataset, labels, attribute)/self.calculate_intrinsic_information__(dataset, labels, attribute)
        return gain_ratio


    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current  tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """
        """
            Your implementation
        """
        max_value=-100
        best_attribute=''
        for feature in self.features:
            if feature not in used_attributes:
                if self.criterion=="gain ratio":
                    gain_value=self.calculate_gain_ratio__(dataset,labels,feature)
                elif self.criterion=="information gain":
                    gain_value=self.calculate_information_gain__(dataset,labels,feature)
                if gain_value>max_value:
                    max_value=gain_value
                    best_attribute=feature

        new_node=TreeNode(best_attribute)        
        index = np.where(np.array(self.features) == best_attribute)[0][0]
        used_attributes.append(best_attribute)
        full_data = np.column_stack((dataset, labels))
        column = full_data[:, index]
        unique, counts = np.unique(column, return_counts=True)
        for value in unique:
            filtered_dataset = full_data[full_data[:,index] == value]
            filtered_labels = filtered_dataset[:,-1]
            unique1 = np.unique(filtered_labels)
            unique1=unique1.astype(int)
            if len(unique1) == 1:
                 leaf = TreeLeafNode(None, unique1[0])
                 new_node.subtrees[value] = leaf
            else:
                new_node.subtrees[value] = self.ID3__(filtered_dataset, filtered_labels, used_attributes)     
        return new_node


    def predict(self, x):
        """
        :param x: a data instance, 1 dimensional Python array 
        :return: predicted label of x
        
        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label
        """
        """
            Your implementation
        """
        current_node=self.root
        while (isinstance(current_node,TreeLeafNode)==False):
            index = np.where(np.array(self.features) == current_node.attribute)[0][0]
            value=x[index]
            current_node=current_node.subtrees[value]

        return current_node.labels

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print(self.root.subtrees)
        print(self)
        print("Training completed")