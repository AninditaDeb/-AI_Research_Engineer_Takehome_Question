import numpy as np
import pandas as pd
import json
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
dib = pd.read_csv('diabetes.csv') 
dib_mod=dib.loc[:,['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']]
X=dib_mod[['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age']]
Y=dib_mod['Outcome']
scaler=MinMaxScaler()
X=scaler.fit_transform(X)
classifier=LogisticRegression()
classifier.fit(X, Y)
Y_pred = classifier.predict(X)
Y_pred_prob = classifier.predict_proba(X)
Y_pred_prob0 = Y_pred_prob[: ][: , 0]
Y_pred_prob1 = Y_pred_prob[: ][: , 1]
N=20
x_outputs=[]
t_labels=[]
for i in range(N):
    x_outputs.append([float(Y_pred_prob0[i]),float(Y_pred_prob1[i])])
    t_labels.append(int(Y[i]))
my_dict = {
    "metric" : ["Accuracy","Sensitivity","Specificity","AUC","SensitivitySpecificityEquivalencePoint"],
    "model_outputs" : x_outputs,
    "gt_labels" : t_labels,
    "threshold" : 0.8,
    "ci": True,
    "num_bootstraps":200,
    "alpha":5
}
with open(sys.argv[1], 'w') as file:
    json.dump(my_dict, file)
