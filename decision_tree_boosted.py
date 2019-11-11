
import numpy as np
import pandas as pd
import decision_tree
import cross_vali
import data_preprocess as dp


def boosted_classify_dt(x, y, test, parameter):
  train = np.concatenate((np.array(x),np.array([y]).T), axis=1)
  train = pd.DataFrame(train)
  test = pd.DataFrame(test)
  test['label'] = 0 # here I fill the label column with dummmy values to make its dimension the same as the training set

  k = 5 # boosting rounds
  weight = np.full(len(x), 1 / len(x))
  boosting_sum = np.zeros((k,len(test)))
  #alpha = []
  #classification = np.zeros(k, len(test))
  
  # print("weight: ", weight)
  classifier_weight = []
  for i in range(k):
    
    # print("i=", i)
    sample_indices = np.random.choice(len(train), size=len(train), replace=True, p=weight)
    # print("samples index ", sample_indices)
    bootstarpped_train = train.iloc[sample_indices, :]
    bootstarpped_result = train.iloc[sample_indices, -1]
    bootstarpped_weight = [weight[x] for x in sample_indices]
    
    # print("samples weight ", bootstarpped_weight)
    dt = decision_tree.create_tree(bootstarpped_train)
    # print("decision tree: ", dt)
    result = train.apply(lambda x: decision_tree.predict(x, dt), axis=1)
    result_test = test.apply(lambda x: decision_tree.predict(x, dt), axis=1)
    # print("predict result: ", list(result.values))
    # print("actual result: ", list(y))
    

    miss = np.logical_xor(result.values, y)
    # print("miss: ", miss)
    error = np.sum(np.multiply(weight, miss)) / np.sum(weight)
    # print("error: ", error)
    if (error > 0.5):
      pass
    cw = 0.5 * np.log((1 - error) / error)

    # print("cw ",cw)
    boosting_sum[i,:] = np.multiply([float(1 if r > 0 else -1) for r in result_test],cw)
    # print("sum: ", list(boosting_sum))


    weight = np.multiply(weight, np.exp([float(1 if m > 0 else -1) * cw for m in miss]))
    weight = [float(i) / sum(weight) for i in weight]
    # print("updated weight: ", weight)

  classification = np.sign(boosting_sum.sum(axis=0))
  # print(classification) 
  classification = [1 if c > 0 else 0 for c in classification]
  return classification
  
  
if __name__ == "__main__":
  x, y = dp.data_read("project3_dataset1.txt")
  cross_vali.cross_validation(x, y, boosted_classify_dt, None)