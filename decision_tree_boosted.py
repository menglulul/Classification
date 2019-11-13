
import numpy as np
import pandas as pd
import decision_tree
import cross_vali
import data_preprocess as dp

def sample(data, weight):
    sample_indices = np.random.choice(len(data), size=len(data), replace=True, p=weight)
    bootstrap_x = data.iloc[sample_indices, :] # sampled training x
    bootstrap_y = data.iloc[sample_indices, -1] # sampled training y
    bootstrap_weight = [weight[s] for s in sample_indices] # sampled training weight
    return bootstrap_x, bootstrap_y, bootstrap_weight

def classify_boosting(train, test, n_rounds):

  weight = np.full(len(train), 1 / len(train)) # weight for all records
  boosting_sum = np.zeros((n_rounds, len(test))) # sum matrix for all rounds
  restart = False
  
  for i in range(n_rounds):
    # for each round, sample the training set with replacement according to weight
    bootstrap_train_x, bootstrap_train_y, bootstrap_weight = sample(train, weight)
    # generate decision tree
    dt = decision_tree.create_tree(bootstrap_train_x, 0, 3)
    # apply decision tree on training dataset
    result_train = train.apply(lambda r: decision_tree.predict(r, dt), axis=1)
    # apply decision tree on testing dataset    
    result_test = test.apply(lambda r: decision_tree.predict(r, dt), axis=1)
    # miss = 1 if misclassified else 0
    miss = np.logical_xor(result_train.values, bootstrap_train_y)
    # error = sum(miss(i)*weight(i))/sum(weight(i))
    error = np.sum(np.multiply(weight, miss)) / np.sum(weight)
    # if error > 0.5 then start over
    if (error > 0.5):
      restart = True
      break
    # alpha(classifier weight) =  1/2 * ln(1- error / error).
    alpha = 0.5 * np.log((1 - error) / error)
    # calculate sum of alpha * y_test for this round
    boosting_sum[i,:] = np.multiply([float(1 if r > 0 else -1) for r in result_test], alpha)
    # update weight
    # new weight = e ^ (alpha * miss)
    weight = np.multiply(weight, np.exp([float(1 if m > 0 else -1) * alpha for m in miss]))
    # normalize weight
    weight = [float(w) / sum(weight) for w in weight]
  if not restart:
    # get final prediction based on the sum of weighted prediction of all rounds
    classification = np.sign(boosting_sum.sum(axis=0))
    classification = [1 if c > 0 else 0 for c in classification] # convert -1s to 0s
    return classification
  else:
    return classify_boosting(train, test, n_rounds)

def boosting(x, y, test, n_rounds):
  train = np.concatenate((np.array(x),np.array([y]).T), axis=1)
  train = pd.DataFrame(train)
  test = pd.DataFrame(test)
  test['label'] = 0 # here I fill the label column with dummmy values to make its dimension the same as the training set
  return classify_boosting(train, test, n_rounds)

  
if __name__ == "__main__":
  x, y = dp.data_read("project3_dataset1.txt")
  cross_vali.cross_validation(x, y, boosting, 2)