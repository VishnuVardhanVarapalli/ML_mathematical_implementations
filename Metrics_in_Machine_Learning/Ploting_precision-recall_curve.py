import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn

dataset = pandas.read_csv("Metrics_in_Machine_Learning\\loan_modified.txt",sep = ",")

test = dataset[dataset['source']!="train"]
train = dataset[dataset['source']=="train"]

X_train = train.drop(["loan_status","source","Loan_ID"],axis = 1)
y_train = train.loan_status

X_test = test.drop(["loan_status","source","Loan_ID"],axis = 1)
y_test = test.loan_status

model = sklearn.linear_model.LogisticRegression()

model.fit(X_train.values,y_train.values)

y_prob = model.predict_proba(X_test.values)

threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def true_positive(y_true, y_pred):
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp

def true_negative(y_true, y_pred):
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn

def false_positive(y_true, y_pred):
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp

def false_negative(y_true, y_pred):
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn

def precision(y_test, y_pred):
    tp = true_positive(y_test, y_pred)
    fp = false_positive(y_test, y_pred)
    if(tp==0 & fp==0):
        return 0
    else:
        return(tp/(tp+fp))
    
def recall(y_test, y_pred):
    tp = true_positive(y_test, y_pred)
    fn = false_negative(y_test, y_pred)
    if(tp==0 & fn==0):
        return 0
    else:
        return(tp/(tp+fn))

precisions = []
recalls = []

for i in threshold_list:
    k = y_prob[:,1] > i
    j = []
    for i in k:
        j.append(int(i))
    precisions.append(precision(y_test, j))
    recalls.append(recall(y_test, j))

plt.figure(figsize = (7,7))
plt.fill_between(precisions, recalls, alpha=0.4)
plt.plot(precisions, recalls, lw = 2)
plt.title("Precision-Recall Curve")
plt.xlim(0,1.0)
plt.ylim(0,1.0)
plt.xlabel("precisions",fontsize = 16)
plt.ylabel("recalls",fontsize = 16)
plt.show()