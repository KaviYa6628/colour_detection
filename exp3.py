

import csv
import random
import math
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
import warnings

warnings.filterwarnings("ignore")
# Step 1: Convert TXT to CSV
import re 
with open('heartdisease.txt', 'r') as infile:
    lines = [re.split(r'[, \s]+',line.strip()) for line in infile if line.strip()]

with open('heartdisease.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['age', 'sex', 'cp', 'restbp', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'])
    writer.writerows(lines)

# Step 2: Read CSV
filename = 'heartdisease.csv'
with open(filename, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)[1:]

dataset = [[float(x) for x in row] for row in data]

# Functions to compute mean and std deviation
def mean(values): return sum(values) / float(len(values))
def stdev(values):
    if len(values)<= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((x - avg) ** 2 for x in values) / float(len(values) - 1))

# Train/Test split
train_size = int(0.75 * len(dataset))
train_set = random.sample(dataset, train_size)
test_set = [row for row in dataset if row not in train_set]

# Organize by class
def separate_by_class(data):
    classes = {}
    for row in data:
        label = row[-1]
        if label not in classes:
            classes[label] = []
        classes[label].append(row)
    return classes

# Summarize dataset
def summarize_dataset(data):
    summaries = [(mean(col), stdev(col)) for col in zip(*data)]
    del summaries[-1]  # remove class label
    return summaries

class_data = {}
for class_value, rows in separate_by_class(train_set).items():
    class_data[class_value] = summarize_dataset(rows)

# Naive Bayes prediction
def calculate_probability(x, mean, stdev):
    if stdev == 0: return 1.0
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, row):
    probabilities = {}
    for class_value, class_summ in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summ)):
            mean_, stdev_ = class_summ[i]
            x = row[i]
            probabilities[class_value] *= calculate_probability(x, mean_, stdev_)
    return probabilities

def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    return max(probabilities, key=probabilities.get)

# Predict test set
y_test = [row[-1] for row in test_set]
y_pred = [predict(class_data, row) for row in test_set]

# Accuracy
correct = sum(1 for i in range(len(y_test)) if y_test[i] == y_pred[i])
accuracy = correct / len(y_test) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

# F1 Score and Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nF1 Score:")
print(f1_score(y_test, y_pred, average='weighted'))

# ROC Curve plotting
y_true_bin = np.zeros((len(y_test), 5))
y_pred_bin = np.zeros((len(y_pred), 5))
for i in range(len(y_test)):
    y_true_bin[i][int(y_test[i])] = 1
    y_pred_bin[i][int(y_pred[i])] = 1

fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 5
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Micro and macro averages
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average (area = {0:0.2f})'.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average (area = {0:0.2f})'.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'orange', 'cornflowerblue', 'red', 'black'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class')
plt.legend(loc="lower right")
plt.show()



NB_from_Gaussian_Sklearn.py
