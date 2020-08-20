import os
import sys

dataset = sys.argv[1]

os.system('python3 get_train_labels.py ' + dataset)
os.system('python3 train_predict_metamodels.py ' + dataset)
os.system('python3 ensemble_classifiers.py ' + dataset)
# 
# file = open('done_datasets.txt', 'a')
# file.write(dataset + '\n')
# file.close()
