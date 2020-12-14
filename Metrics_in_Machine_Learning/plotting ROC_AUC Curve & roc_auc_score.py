import pandas 
import sklearn.linear_model
import sklearn.tree
import matplotlib.pyplot as plt
import sklearn.ensemble
import sklearn.metrics

dataset = pandas.read_csv("C:\\Users\\iamvi\\OneDrive\\Desktop\\New Text Document (2).txt",sep = ',')

test = dataset[dataset['source']!="train"]
train = dataset[dataset['source']=="train"]

X_train = train.drop(["loan_status","source","Loan_ID"],axis = 1)
y_train = train.loan_status

X_test = test.drop(["loan_status","source","Loan_ID"],axis = 1)
y_test = test.loan_status

#model = sklearn.linear_model.LogisticRegression()
#model = sklearn.tree.DecisionTreeClassifier()
model = sklearn.ensemble.RandomForestClassifier()
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
 
tpr = []
fpr = []
for i in threshold_list:
    k = y_prob[:,1] > i
    j = []
    for i in k:
        j.append(int(i))
    tp = true_positive(y_test,j)
    fp = false_positive(y_test,j)
    tn = true_negative(y_test,j)
    fn = false_negative(y_test,j)
    tpr.append(tp/(tp+fn))
    fpr.append(fp/(fp+tn))
data = list(zip(tpr,fpr))
dataset = pandas.DataFrame(data,columns = ["tpr","fpr"])

plt.figure(figsize = (7,7))
plt.fill_between(dataset.fpr.values,dataset.tpr.values, alpha=0.4)
plt.plot(dataset.fpr.values,dataset.tpr.values,lw = 2)
plt.title("ROC_AUC CURVE")
plt.xlim(0,1.0)
plt.ylim(0,1.0)
plt.xlabel("fpr",fontsize = 16)
plt.ylabel("tpr",fontsize = 16)
plt.show()

print(sklearn.metrics.roc_auc_score(y_test,y_prob[:,1])) #which gives the area of the blue region in plotted curve.
