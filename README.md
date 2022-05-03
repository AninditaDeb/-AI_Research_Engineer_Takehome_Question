#####Problem Statement##################
#AI_Research_Engineer_Takehome_Question
For this question, we ask you to create a small codebase for evaluating the
performance of binary classification models. It should take approximately 1 to 2
hours.
Please send us the link to the GitHub repository of your code and test cases.
We will run the following command line code python main.py $INPUT_JSON
$OUTPUT_JSON
Implement the following standard metrics:
Accuracy
Sensitivity
Specificity
AUC (Area under the receiver operating characteristic curve)
We would also like you to implement a new metric: the point on the ROC curve
where sensitivity and specificity are equal —
SensitivitySpecificityEquivalencePoint . Given a standard ROC curve, this would
be the point at which (1 - x) = y . If that exact point does not exist on
curve, the metric should return the average of the nearest points on the curve.
All relevant metrics should also support computation of a confidence interval
using the percentile bootstrapping method with a given alpha between 0 and 1.
To compute the confidence interval using percentile bootstrapping, you
need to draw K samples with replacement from the data. For each of those
K samples, you can compute the metrics of interest. This will give K
results. The confidence interval for the metrics at level alpha is
defined as the $\alpha$th and $(1-\alpha)$th percentile of these K
results.
Create a main.py file which inputs a single JSON file and outputs a single JSON
file.
for example, python main.py example_input.json example_output.json
The input JSON keys and value types are as follows:
{
'metric': str, # ‘Accuracy’, ‘Sensitivity’, ‘Specificity’, ‘AUC’, and
‘SensitivitySpecificityEquivalencePoint’.
'model_outputs': list of list of float, # shape Nx2. N examples with
2 outputs each.
'gt_labels': list of int, # shape N. Ground truth class labels of N
examples.
'threshold': float, # Threshold for metrics with operating points (eg
'Accuracy').
'ci': bool, # If True, include confidence interval in output.
'num_bootstraps': int, # Number of iterations of bootstrapping to
compute confidence interval.
'alpha': float # alpha parameter for confidence level of the
interval.
}
The output JSON output.json should have the expect keys and value types:
{
'value': float, # Value of the metric specified in the input JSON's
metric key.
'lower_bound': float, # Lower bound of the confidence interval. Only
required if ci=True in input JSON.
'upper_bound': float # Upper bound of the confidence interval. Only
required if ci=True in input JSON.
}
You are allowed to use the following external libraries:
numpy, scipy, sklearn, and pandas
any of the common python testing libraries

**The process is to run this file in colab or kaggle or jupyter**

**!python main.py --Input example_input.json --Dataset diabetes.csv --Output example_output.json*
**Make sure your input json file metrics keys and the main file calculating the values for those keys should in sync other wise there would be exception thrown and the 
output json file could not be formed**


** I have tested my data with diabetes datset avaliable in Kaggle  https://www.kaggle.com/datasets/mathchi/diabetes-data-set,now since I am using this data for modelling because as mentioned in the input json file we should get the model outputs as one of the parameters so I had to start from somewhere and in order to do so 
i created one script called Actual.py which is taking data from diabetes data,doing the preprocessing stuff and then modelling with Logistic Regression.After the model preditcs the probability of being diabetic and not being diabetic,these data along with other parameters such as ci,metrics all these parameter values are passed on to a a json input file example_input.json.

Now comes the main part , i designed a main.py file as per the requirement which takes the model outputs from the model output parameter passed to the input json file and then calculates sensitivity specificity,AUC and point of equivalence for sensitivity and specificity for a range of threholds.

In case we need to estimate the confidence interval then according to the procedure metioned in the question document I am again feeding data from my dataset to this main.py file since I have to take K bootstrap samples from the data.You cab see if data[ci==True] under this command I have done this process of bootstraping k samples (again the value for k can be changed in the input json file from data for which the confidence interval would be calculated.I have set a range of threholds in my main.py file and for each of this threhold I am taking k bootstraping samples for calculating the lower bound and upperbound for each of the metrics Accuracy,Sensitivity and Specificity.In summary since we have to perform the boot straping times then if we are comparing one metrics say sensitvity for one threhold to another threhold the as per my understanding I should comapre the mean of this metrics obtained for K bootstraped samples obtained at each of the threholds.The main reason of taking a range of threholds was because AUC is calculated for a range of therholds, similarly while checking for point of equivalence for senmsitivity and specificity also I have to consider a range of threholds.However as per the question I am returning the metrics value for only one threhold that is mentioned in the input json file.So if ci=True in my input json file or in other words I am performing K bootstraping then in that case the output json file contains the mean value of the metrics for that particular threshold mentioned in the input json file.I have also handled all kinds of exceptions according to my understanding one of which includes if the the output json file does not contain the reuired number of metrics as is mentioned inthe input json file than it would raise an exception.Also i have included the parser arparser syntax in my main. py file so that while executing the main.py file if you miss passing one of this files as argument the main.py file would pop a message.

**You might want to change a different dataset then you have to change it in the command line it self  well********************.Because bootstraping needs the actual data however since I have performed some preprocessing on the data in my main.py file :
dataset=dataset.loc[:,['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']]
dataset1=np.array(dataset)
data_new=dataset1[:,(1,2,3,4,5)]
target=dataset1[:,6],

**Please note:If you wish to remove the above part you can do so,but according to my script the test and target needs to be kept separated**
