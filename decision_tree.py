
import numpy as np
import pandas as pd
import cross_vali
import data_preprocess as dp

def classify_data(data):
  labels, n_labels = np.unique(data.iloc[:, -1], return_counts=True)
  max_i = np.argmax(n_labels)
  return len(labels), labels[max_i]

def get_possible_splits(data):
  splits = []
  n_rows, n_columns = data.shape
  for i in range(n_columns - 1):
    col = data.iloc[:,i]
    splits.append(np.unique(col))
  return splits

def split_data(data, column, value):
  split_column = data.iloc[:, column]
  left = data[split_column <= value]
  right = data[split_column > value]
  return left, right

def calculate_gini(left, right):
  p_left = len(left) / (len(left) + len(right))
  p_right = len(right) / (len(left) + len(right))
  c1_left = 0
  c0_left = 0
  c1_right = 0
  c0_right = 0
  if (len(left) > 0):
    c1_left = len(left[left.iloc[:, -1] == 1]) / len(left) 
    c0_left = len(left[left.iloc[:, -1] == 0]) / len(left)
  gini_left = 1 - pow(c1_left, 2) - pow(c0_left, 2)
  if (len(right) > 0):
    c1_right = len(right[right.iloc[:, -1] == 1]) / len(right)
    c0_right = len(right[right.iloc[:, -1] == 0]) / len(right)
  gini_right = 1 - pow(c1_right, 2) - pow(c0_right, 2)
  
  gini = p_left * gini_left + p_right * gini_right

  return gini
  
def choose_split(data, possible_splits):
  min_gini = 1
  best_column = 0
  best_value = 0
  for i in range(len(possible_splits)):
    col = possible_splits[i]
    for val in col:
      left, right = split_data(data, i, val)
      gini = calculate_gini(left, right)
      if gini < min_gini:
        min_gini = gini
        best_column = i
        best_value = val
  return best_column, best_value

def create_tree(data):
  
  # If all items in the dataset have the same classification, return the classification.
  n_labels, classification = classify_data(data)
  if (n_labels == 1):
    return classification

  # recursively generate subtrees
  else:
    possible_splits = get_possible_splits(data)
    best_col, best_val = choose_split(data, possible_splits)
    question = "{} <= {}".format(best_col, best_val)
    tree = {}
    tree[question] = []
    data_left, data_right = split_data(data, best_col, best_val)
    tree_left = create_tree(data_left)
    tree_right = create_tree(data_right)
    tree[question].append(tree_left)
    tree[question].append(tree_right)
    return tree
    
def predict(row, tree):
  question = list(tree.keys())[0]
  attr, operator, val = question.split(" ")
  if (row[int(attr)] <= float(val)):
    subtree = tree[question][0]
  else:
    subtree = tree[question][1]
  if not isinstance(subtree, dict):
    return subtree
  else:
    return predict(row, subtree)
  
def classify_dt(x, y, test, parameter):
  train = np.concatenate((np.array(x),np.array([y]).T), axis=1)
  train = pd.DataFrame(train)
  test = pd.DataFrame(test)
  test['label'] = 0 # here I fill the label column with dummmy values to make its dimension the same as the training set
  dt = create_tree(train)
  # print("decision tree: ", dt)
  # print(test.apply(lambda x: predict(x, dt), axis=1))
  return test.apply(lambda x: predict(x, dt), axis=1)

  # todo: max_depth - regularization

if __name__ == "__main__":
  x, y = dp.data_read("project3_dataset1.txt")  
  cross_vali.cross_validation(x, y, classify_dt, None)