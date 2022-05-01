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
