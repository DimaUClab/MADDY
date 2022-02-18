# MADDY Classification

Multiclass Classification Analysis and Quality Testing of MADDY (Microtubule Assembly and Disassembly Dynamics model) Output

Published by: Maria Kelly - Dima Group @ University of Cincinnati

Details of applied analysis can be found in original publication: https://doi.org/10.1016/j.csbj.2022.01.028

Overview: 
This script will give a multiclass classification of a given dataset by applying cross validation to several classification and ensemble algorithms in order to compare the predicition accuracy and standard deviation of each algorithm.  The SMOTE oversampling technique will be applied in case of uneven class populations. Further quality analysis (confusion matrices, PR, and ROC curves) will be made for selected alorithms of interest.

Procedure:
  1) User Input: Enter the name of the file containing all features and class labels in the order as they appear in dataset (the header of class labels will be needed in order for the script to identify the correct column).  Enter class names and a selection of colors equal to the length of classes for plot parameterization. "clf_dict" is a created dictionary of the selection of algorithms found to have the best initial prediction.
  2) Loading Data: Script will attempt to identify number of features automatically. A prompt will allow the users to continue with the classification if the feature count is correct or if a number of features should be removed.  Enter feature names to be removed in prompted section to have a new dataframe be generated.
  3) Feature Distribution: The distribution of each feature will be generated on one saved figure.
  4) Classification: A repeated stratified kfold was utlized due to the skew found in the datasets of original publication. A list of classifiers used and explanations for parameters used is describe in detail in the original publication. SMOTE was also tested using another selection of methods.  Two boxplots representing predicition accuracy measurments will be generated.
  5) Confusion Matrices: Multiclass confusion matrices will be generated and label based off of the 'class_names' variable listed above. Center value for each unit describe the sum of testing set predictions for all kfolds. Additional values included for the total number of predictions per true class and percentage of true class predictions that unit represents. The colorbar represents strength of prediction success
  6) PR/ROC Curves: Using the One vs Rest approach, the testing set of the last kfold will be plotted to compare the selection of algorithms listed in 'clf_dict'.
