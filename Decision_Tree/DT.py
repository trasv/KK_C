import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from statistics import mean
import warnings
warnings.filterwarnings('ignore')
import csv
import math
from random import randrange
from random import seed

ys = pd.read_csv('yeast.data', sep = '\s+' ,header = None)
ys.columns=['id','mcg','gvh','alm','mit','erl','pox','vac','nuc','class']
print(ys)
yt = ys.iloc[:,1:]
print(yt)

yt['Class']=yt['class']
yt['Class'] = yt['Class'].map({'MIT': 0, 'NUC': 1, 'CYT': 2, 'ME1': 3, 'EXC': 4, 'ME2': 5, 'ME3': 6, 'VAC': 7, 'POX': 8, 'ERL': 9})
yt["Class"].unique()
yt = yt.drop(['class'],axis=1)
yt.info()
dataset = yt

def cross_validation(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)/n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def accuracy_metric(actual,predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct+=1
    return correct/float(len(actual)) * 100.0

def evaluate_algorithm(dataset, algorithm, n_folds, method, *args):
    folds = cross_validation(dataset, n_folds)
    scores = list()
    score_recall = list()
    score_precision = list()
    x = 1
    y = 0
    for fold in folds:
        print('Fold ke-%d' % x)
        train_set = list(folds)
        train_set.pop(y)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = decision_tree(train_set, test_set, method, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        rec = recall(actual, predicted)
        pre = precision(actual, predicted)
        score_recall.append(rec)
        score_precision.append(pre)
        scores.append(accuracy)
        x+=1
        y+=1
    return scores, score_recall, score_precision

def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    
    return left, right

def gini_index(groups, classes):
    n_instances= float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val)/size
            score += p * p
        gini += (1.0 - score) * (size/n_instances)
    return gini

def entropi_parent(dataset):
    kelas = []
    for row in dataset:
        kelas.append(row[-1])
    class_values = list(set(row[-1] for row in dataset))
    score = 0.0
    size = float(len(kelas))
    for class_val in class_values:
        p = [row[-1] for row in dataset].count(class_val)/size
        if p!=0:
            score += (-1 * p * math.log2(p))
        else:
            score=score
    return score
    
def entropi(groups, classes, parent):
    n_instances= float(sum([len(group) for group in groups]))
    entropi = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val)/size
            if p!=0:
                score += (-1 * p * math.log2(p))
            else:
                score=score
        entropi += score * (size/n_instances)
    return (parent-entropi)  

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999,999,999, None
    b_scoremax = -1
    eparent = entropi_parent(dataset);
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            if method == 1:
                gini = gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini , groups
            elif method == 2:
                en = entropi(groups, class_values, eparent)
                if en > b_scoremax:
                    b_index, b_value, b_scoremax, b_groups = index, row[index], en , groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}
   
def loadData(filename, dataset):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        data = list(lines)
        for x in range(len(data)-1):
            for y in range(8):
                data[x][y] = float(data[x][y])
            dataset.append(data[x])
            
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

def build_tree(train, max_depth, min_size):
    	root = get_split(train)
    	split(root, max_depth, min_size, 1)
    	return root

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
        
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    print_tree(tree)
    return predictions

def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s [%s]' % ((depth*' ', node)))

yst = yt.to_numpy()

seed(1)
dataset = []
method = int(input('1. Gini\n2. Entropi\n'))
n_folds = 3 # diganti ganti sesuai k-fold nya
max_depth = 3
min_size = 10
scores, score_recall, score_precision = evaluate_algorithm(yst, decision_tree, n_folds, max_depth, min_size)
print('Mean Accuracy : %.2f%%' % (sum(scores)/float(len(scores))))
print('Mean Recall : %.2f%%' % (sum(score_recall)/float(len(score_recall))))
print('Mean Precision : %.2f%%' % (sum(score_precision)/float(len(score_precision))))