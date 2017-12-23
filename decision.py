from pprint import pprint
from scipy.stats import chi2, chi2_contingency
from sys import argv
import numpy as np
import pandas as pd


file_name = argv[1]

# alpha values for chi square pruning
if len(argv) > 2:
    alpha = float(argv[2])
else:
    alpha = 0.05  # 5%



def partition(a):
    """takes values of an attribute as parameter and splits it based on values

    params:
        a (list):
        list of values of attribute

    return:
         (dict):
         dictionary where keys are various values of attribute and value is array of position where the values occurs
    """
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}


def entropy(s):
    """takes values of an attribute as parameter and returns entropy

    params:
        s (list):
        list of values of attribute

    return:
         res (float):
         Entropy value of the attribute
    """
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float')/len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


def information_gain(y, x):
    """takes values of an attribute as parameter and returns entropy

    params:
        y (list):
        list of values of class
        x (list):
        list of values of attribute

    return:
         res (float):
         Information gain of the attribute
    """

    res = entropy(y)

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res


def is_pure(s):
    """takes values of class as parameter and returns if all have same value

    params:
        s (list):
        list of values of class

    return:
         (boolean):
         True if pure else false
    """

    return len(set(s)) == 1


def recursive_split(x, y, attr_name):
    """takes examples and values of class as parameter and returns best split of the data

    params:
        x (list of list):
        list of examples having various attributes
        y (list)
        list of class values of the examples
        attr_name (list of string)
        list of names of the attributes

    return:
         (dictionary):
         key is the best split with max information gain and values is either a class or a dictionary (decision tree)
    """
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return str(y[0])

    # We get attribute that gives the highest mutual information
    gain = np.array([information_gain(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        return y

    # We split using the selected attribute
    sets = partition(x[:, selected_attr])

    res = {}
    for k, v in sets.items():
        # create subsets of data based on the split
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)
        x_subset = np.delete(x_subset, selected_attr, 1)
        attr_name_subset = attr_name[:selected_attr] + attr_name[selected_attr+1:]
        #recurse on subset of data left
        res[attr_name[selected_attr] + " = " + k ] = recursive_split(x_subset, y_subset, attr_name_subset)

    return res


def pruneLeaves(obj):
    """takes decision tree as parameter and returns a pruned tree based on chi square

    params:
        obj (dict):
        obj is a decision tree encoded in the form of decision tree

    return:
         obj (dict):
         obj is decision tree with pruned leaves
    """
    isLeaf = True
    parent = None
    for key in obj:
        if isinstance(obj[key], dict):
            isLeaf = False
            parent = key
            break
    if isLeaf and obj.keys()[0].split(' ')[0] not in satisfied_attributes:
        global pruned
        pruned = True
        return 'pruned'
    if not isLeaf:
        if pruneLeaves(obj[parent]):
            obj[parent] = None
    return obj


#read examples from the csv file
data = np.loadtxt(open(file_name, "rb"), delimiter=",", dtype='string', converters = {3: lambda s: s.strip()})
#get first name for the attribute name
attr_name = data.take(0,0)[:-1].tolist()
#get last column for class attribute value
y = data[...,-1][1:]
#get rest of the data for the examples
X = data[...,:-1]
X = np.delete(X,0,0)

#call recursive_split to train the decision tree
tree = recursive_split(X, y, attr_name)

satisfied_attributes = []
for i in range(10):
    contengency = pd.crosstab(X.T[i], y)
    c, p, dof, expected = chi2_contingency(contengency)
    if c > chi2.isf(q=alpha, df=dof):
        satisfied_attributes.append(attr_name[i])

print '\nDecision tree before pruning-\n'
pprint(tree)

print '\nDecision tree after pruning-\n'
pruned = True
while pruned:
    #keep pruning till leaf nodes can be pruned or till whole tree has been pruned
    pruned = False
    tree = pruneLeaves(tree)
    if tree == 'pruned':
        break
pprint(tree)
