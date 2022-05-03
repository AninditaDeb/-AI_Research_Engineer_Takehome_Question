#!/usr/bin/env python
# coding: utf-8

# Author : Anindita Deb
# Task :AI_Research_Engineer_Takehome_Question
# Date :05/03/2022
# Identity: Masters Student at State University of New York at Buffalo
# Task Desc: Performance metrics,Sensitivity,Specificity,AUC,SensitivitySpecificityEquivalencePoint and Confidence Interval of these    statistics ##############################


#################################Imported the main libraries################################################################
import json
import sys
import numpy as np
import pandas as pd
from numpy.random import seed
from numpy.random import randint
from numpy.random import rand
from sklearn.metrics import confusion_matrix,auc
from numpy import percentile
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import argparse

parser = argparse.ArgumentParser(description='Input and output file')
parser.add_argument('--Input', dest='Input', type=str, help='Name of input json file',required=True)
parser.add_argument('--Dataset', dest='Dataset', type=str, help='File path and file name of the dataset',required=True)
parser.add_argument('--Output', dest='Output', type=str, help='Name of output json file',required=True)

args = parser.parse_args()
####################################Loading the json file  with input data and #############################################
########checking for the boolean value ci mentioned in input json file######################################################
class User_Error(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return(repr(self.value))       
with open(args.Input, 'r') as f:
    data = json.load(f)
    keys=list(data.keys())
    try:
        actual_data=["metric","model_outputs","gt_labels","threshold","ci","num_bootstraps","alpha"]
        for i in actual_data:
            if i not in data.keys():
                raise(User_Error("One of the key inputs {} is missing from input json file {}".format(i,args.Input)))
            elif keys.index(i)!=actual_data.index(i):
                raise(User_Error("The input key {} mentioned in input json file{} is not in desired order ".format(i,args.Input)))
    except User_Error as error:
        print('A New Exception occured:',error.value)
    else:
        if data["ci"]==False:
            model_outputs=pd.DataFrame(data["model_outputs"])
            y_predicted_prob=model_outputs[1]
####################Selecting a range of thresholds#########################################################################
###################### and calculating the Accuracy,Sensitivity and Specificity and AUC ####################################
#########################and SensitivitySpecificityEquivalencePoint#########################################################
            numbers = [float(x)/10 for x in range(10)]
            y_data=pd.DataFrame()
            y_actual_Diabetes=pd.DataFrame(data["gt_labels"])
            y_data["actual_Diabetes"]=y_actual_Diabetes
            y_data["y_predicted_prob"]=model_outputs[1]
            cutoff_df=pd.DataFrame( columns = ['Threshold','Accuracy','Sensitivity','Specificity'])
            for i in numbers:
                y_data[i]= y_data.y_predicted_prob.map(lambda x: 1 if x > i else 0)
            for i in numbers:
                cm1 = confusion_matrix(y_data.actual_Diabetes, y_data[i])
                total1=sum(sum(cm1))
                Accuracy = (cm1[0,0]+cm1[1,1])/total1
                Specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
                Sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
                cutoff_df.loc[i]=[ i ,Accuracy,Sensitivity,Specificity]
  
            cutoff_df.reset_index(drop=True,inplace=True)
            AUC=auc(1-cutoff_df["Specificity"],cutoff_df["Sensitivity"])
            fig, ax = plt.subplots(1, figsize=(8, 6))
            fig.suptitle('Sensitivity and Specificity Plot', fontsize=15)
            ax.plot(cutoff_df["Threshold"], cutoff_df["Sensitivity"], color="green", label="Sensitivity")
            ax.plot(cutoff_df["Threshold"], cutoff_df["Specificity"], color="blue", label="Specificity")
            ax.plot(cutoff_df["Threshold"], cutoff_df["Accuracy"], color="red", label="Accuracy")
            plt.legend(loc="upper right")
            plt.show()
            x = cutoff_df["Threshold"]
            y1 = cutoff_df["Sensitivity"]
            y2 = cutoff_df["Specificity"]
            slope_intercept1 = np.polyfit(x,y1,1)
            slope_intercept2 = np.polyfit(x,y2,1)
            xi = (slope_intercept1[1]-slope_intercept2[1]) / (slope_intercept2[0]-slope_intercept1[0])
            SensitivitySpecificityEquivalencePoint = slope_intercept1[0] * xi + slope_intercept1[1]
            lower_bound={"lower_bound_Accuracy":"NA","lower_bound_Sensitivity":"NA","lower_bound_Specificity":"NA"}
            upper_bound={"upper_bound_Accuracy":"NA","upper_bound_Sensitivity":"NA","upper_bound_Specificity":"NA"}
        
        else:
            dataset=pd.read_csv(args.Dataset) 
            dataset=dataset.loc[:,['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']]
            dataset1=np.array(dataset)
            data_new=dataset1[:,(1,2,3,4,5)]
            target=dataset1[:,6]
            numbers = [float(x)/10 for x in range(10)]
            cutoff_df=pd.DataFrame( columns = ['Threshold','Accuracy','lower_accuracy','upper_accuracy','Sensitivity','lower_sensitivity','upper_sensitivity','Specificity'
                                           ,'lower_specificity','upper_specificity','lower_p','upper_p'])
            num_bootstraps=data["num_bootstraps"]
############################For a number of thresholds we would be calculating confidence interval##########################
############################Using using percentile bootstrapping############################################################
###############Here we select a particular threshold and for that threshold we calculate ###################################
#########Confidence Interval and associated statistics such as Mean Sensitivity,Mean Specificity and Mean Accuracy##########
########The above steps we perform for a selected range of thresholds#######################################################
################For that each of this threshold then we then calculate AUC and 
            for i in numbers:
                seed(1)
                Accuracy_mean=0
                Specificity_mean=0
                Sensitivity_mean=0
                Accuracy_list= list()
                Specificity_list= list()
                Sensitivity_list=list()
                for _ in range(num_bootstraps):
                    indices = randint(0, 700, 500)
                    sample_data  = data_new[indices]
                    sample_target = target[indices]
                    scaler=MinMaxScaler()
                    sample_data=scaler.fit_transform(sample_data)
                    classifier=LogisticRegression()
                    classifier.fit(sample_data, sample_target)
                    Y_pred = classifier.predict(sample_data)
                    Y_pred_prob = classifier.predict_proba(sample_data)
                    Y_pred_prob0 = Y_pred_prob[: ][: , 0]
                    Y_pred_prob1 = Y_pred_prob[: ][: , 1]
                    y_data=pd.DataFrame()
                    y_actual_Diabetes=pd.DataFrame(sample_target)
                    y_data["actual_Diabetes"]=y_actual_Diabetes
                    y_data["y_predicted_prob"]= Y_pred_prob1
                    y_data[i]= y_data.y_predicted_prob.map(lambda x: 1 if x > i else 0)
                    cm1 = confusion_matrix(y_data.actual_Diabetes, y_data[i])
                    total1=sum(sum(cm1))
                    Accuracy = (cm1[0,0]+cm1[1,1])/total1
                    Specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
                    Sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
                #cutoff_df.loc[i]=[ i ,Accuracy,Sensitivity,Specificity]
                    Accuracy_mean+=Accuracy
                    Specificity_mean+=Specificity
                    Sensitivity_mean+=Sensitivity
                    Accuracy_list.append(Accuracy)
                    Specificity_list.append(Specificity)
                    Sensitivity_list.append(Sensitivity)
############We are calculating lower bound and upper bound of all the statistics Accuracy,Sensitivity,Specificity###########
############However for the time being we just refrained from calculating the upperbound and lower bound for################
##############for metrics such as AUC and SensitivitySpecificityEquivalencePoint--????Need Clarification####################

                alpha = data["alpha"]
                lower_p = alpha / 2.0
                lower_accuracy = max(0.0, percentile(Accuracy_list, lower_p))
                lower_Specificity = max(0.0, percentile(Specificity_list, lower_p))
                lower_Sensitivity = max(0.0, percentile(Sensitivity_list, lower_p))
                upper_p = (100 - alpha) + (alpha / 2.0)
                upper_accuracy = min(1.0, percentile(Accuracy_list, upper_p))
                upper_Specificity= min(1.0, percentile(Specificity_list, upper_p))
                upper_Sensitivity= min(1.0, percentile(Sensitivity_list, upper_p))
                Accuracy_mean=(Accuracy_mean/100)
                Specificity_mean=(Specificity_mean/100)
                Sensitivity_mean=(Sensitivity_mean/100)
                cutoff_df.loc[i]=[ i ,Accuracy_mean,lower_accuracy,upper_accuracy,
                              Sensitivity_mean,lower_Sensitivity,upper_Sensitivity,
                              Specificity_mean,lower_Specificity,upper_Specificity,lower_p,upper_p]
            cutoff_df.reset_index(drop=True,inplace=True)
            AUC=auc(1-cutoff_df["Specificity"],cutoff_df["Sensitivity"])
            fig, ax = plt.subplots(1, figsize=(8, 6))
            fig.suptitle('Sensitivity and Specificity Plot', fontsize=15)
            ax.plot(cutoff_df["Threshold"], cutoff_df["Sensitivity"], color="green", label="Sensitivity")
            ax.plot(cutoff_df["Threshold"], cutoff_df["Specificity"], color="blue", label="Specificity")
            ax.plot(cutoff_df["Threshold"], cutoff_df["Accuracy"], color="red", label="Accuracy")
            plt.legend(loc="upper right")
            plt.show()
            x = cutoff_df["Threshold"]
            y1 = cutoff_df["Sensitivity"]
            y2 = cutoff_df["Specificity"]
            slope_intercept1 = np.polyfit(x,y1,1)
            slope_intercept2 = np.polyfit(x,y2,1)
            xi = (slope_intercept1[1]-slope_intercept2[1]) / (slope_intercept2[0]-slope_intercept1[0])
            SensitivitySpecificityEquivalencePoint = slope_intercept1[0] * xi + slope_intercept1[1]
        #print(cutoff_df)
            lower_bound_Accuracy=cutoff_df['lower_accuracy'].loc[cutoff_df.index[int(cutoff_df[cutoff_df['Threshold'] ==data['threshold']].index.values[0])]]
            lower_bound_Sensitivity=cutoff_df['lower_sensitivity'].loc[cutoff_df.index[int(cutoff_df[cutoff_df['Threshold'] ==data['threshold']].index.values[0])]]
            lower_bound_Specificity=cutoff_df['lower_specificity'].loc[cutoff_df.index[int(cutoff_df[cutoff_df['Threshold'] ==data['threshold']].index.values[0])]]
            lower_bound={"lower_bound_Accuracy":lower_bound_Accuracy,"lower_bound_Sensitivity":lower_bound_Sensitivity,"lower_bound_Specificity":lower_bound_Specificity}
            upper_bound_Accuracy=cutoff_df['upper_accuracy'].loc[cutoff_df.index[int(cutoff_df[cutoff_df['Threshold'] ==data['threshold']].index.values[0])]]
            upper_bound_Sensitivity=cutoff_df['upper_sensitivity'].loc[cutoff_df.index[int(cutoff_df[cutoff_df['Threshold'] ==data['threshold']].index.values[0])]]
            upper_bound_Specificity=cutoff_df['upper_specificity'].loc[cutoff_df.index[int(cutoff_df[cutoff_df['Threshold'] ==data['threshold']].index.values[0])]]
            upper_bound={"upper_bound_Accuracy":upper_bound_Accuracy,"upper_bound_Sensitivity":upper_bound_Sensitivity,"upper_bound_Specificity":upper_bound_Specificity}
        Accuracy_val=cutoff_df['Accuracy'].loc[cutoff_df.index[int(cutoff_df[cutoff_df['Threshold'] ==data['threshold']].index.values[0])]]
        Sensitivity_val=cutoff_df['Sensitivity'].loc[cutoff_df.index[int(cutoff_df[cutoff_df['Threshold'] ==data['threshold']].index.values[0])]]
        Specificity_val=cutoff_df['Specificity'].loc[cutoff_df.index[int(cutoff_df[cutoff_df['Threshold'] ==data['threshold']].index.values[0])]]
        metrics={"Accuracy":Accuracy_val,"Sensitivity":Sensitivity_val,"Specificity":Specificity_val,"AUC":AUC,"SensitivitySpecificityEquivalencePoint":SensitivitySpecificityEquivalencePoint}          
        my_dict = {"value" : metrics,"lower_bound" : lower_bound,"upper_bound":upper_bound}
    ###################Exception Handling ,the input json file should have all the required metrics as per the question#######
    #################calculated in the main.py file should be in line with that mentioned in the json input file
        try:
            x,y=len(data["metric"]),len(my_dict["value"])
            z=data["metric"]
            w=my_dict["value"]
            p=list(w.keys())
            output_values=["value","lower_bound","upper_bound"]
            l_bound_calculated=my_dict["lower_bound"]
            l_bound_calculated_keys=list(l_bound_calculated.keys())
            u_bound_calculated=my_dict["upper_bound"]
            u_bound_calculated_keys=list(u_bound_calculated.keys())
            l_bound_actual=["lower_bound_Accuracy","lower_bound_Sensitivity","lower_bound_Specificity"]
            u_bound_actual=["upper_bound_Accuracy","upper_bound_Sensitivity","upper_bound_Specificity"]
            for a in output_values:
                if a not in my_dict.keys():
                    raise(User_Error("One of the Metrics {} is not calculated in main file, the output json file cannot be formed".format(a)))      
                elif a=="value":
                    for i in p:
                        if w[i]=="NA" or w[i] is None:
                            raise(User_Error("The metrics {} cannot be Null or NA, the output json file cannot be formed".format(i)))      
                elif a=="lower_bound":
                    for i in l_bound_actual:
                        if i not in l_bound_calculated.keys():
                            raise(User_Error("The metrics {} is missing, the output json file cannot be formed".format(i)))
                elif a=="upper_bound":
                    for i in u_bound_actual:
                        if i not in u_bound_calculated.keys():
                            raise(User_Error("The metrics {} is missing, the output json file cannot be formed".format(i)))
                
            if x!=y:
                for i in z:
                    if i not in w.keys():
                        raise(User_Error("One of the Metrics {} is missing from main file/not calculated, hence the output json file cannot be formed".format(i)))
            elif x==y:
                for i in z:
                    if z.index(i)!=p.index(i):
                        raise(User_Error("The metrics calculated {} does not follow the same order as of input json file , hence the output json file cannot be formed".format(i)))
                    
            elif data["ci"]==True:
                for i in l_bound_calculated_keys:
                    if l_bound_calculated[i] is None or l_bound_calculated[i] =="NA" :
                        raise(User_Error("One of the metrics {} is missing from calculation irrespective of ci being True in input json file, hence the output json file cannot be formed".format(i)))
                    if u_bound_calculated[i]=="NA" or u_bound_calculated[i] is None:
                        raise(User_Error("One of the metrics {} is missing from calculation irrespective of ci being True in input json file, hence the output json file cannot be formed".format(i))) 
  
        except User_Error as error:
            print('A New Exception occured:',error.value)
        else:
            print("Here is your performance metrics - example_output.json")
            with open(args.Output, 'w') as file:
                json.dump(my_dict, file)
            
       
    

