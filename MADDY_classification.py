# MADDY DATA CLASSIFICATION ANALYSIS AND QUALITY TESTING
# Maria Kelly 

import numpy as np
import pandas as pd
import seaborn as sns
import csv
import sys
import scipy.cluster.hierarchy as shc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from scipy.sparse import data
from sklearn import metrics
from numpy import empty, loadtxt, mean, std
from numpy.lib.utils import safe_eval
from pandas import read_csv
from pandas.core.algorithms import mode
from pandas.io.stata import precision_loss_doc
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib import rcParams
from matplotlib.ticker import StrMethodFormatter
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, mean_squared_error, r2_score, precision_recall_curve, roc_curve
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, scale, label_binarize
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from scipy.cluster.hierarchy import dendrogram, weighted
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer
from xgboost import XGBClassifier
from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Universal matplotlib parameters
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 5
plt.rcParams.update({'mathtext.default':  'regular' })
plt.rcParams['font.family'] = "serif"


##################################################### User Edits ###############################################################
file_name = 'all_events.csv'
class_column_name = 'class'
class_names = ['Catastrophe', 'Growth', 'Rescue', 'Shortening']
colors = ['red', 'blue', 'green', 'orange']
clf_dict = {"RF": RandomForestClassifier(n_estimators=1000), "XGB": XGBClassifier()}
################################################################################################################################


#----------------------------------------------------- LOAD DATA -----------------------------------------------------#
print("\nStarting Classification Analysis and Quality Testing...")
print("Loading Data...")

# Pandas dataframe of original .csv 
dataset = pd.read_csv(file_name)

# Find number of features
count = []
for i in dataset.columns:
    try:
        check = int(i) # If column header is int, then it is counted as a feature
        count.append(i)
    except:
        continue

# Create dataframe of desired features
print("Preprocessing Data...\n")
feat_remove = []
while True:
    data = pd.DataFrame()
    for i in count: # Go through list of extracted features
        if len(feat_remove) > 0: # If there are features to remove
            remove_check = 0
            for remove in feat_remove: # Check if feature is needed to be removed
                if int(i) == int(remove):
                    remove_check = 1
                    break
                else:
                    continue
            if remove_check == 0: # If feature should stay
                data[['{}'.format(i)]] = dataset[['{}'.format(i)]]
            else:
                continue
        else: # No features need to be removed
            data[['{}'.format(i)]] = dataset[['{}'.format(i)]]

    data[['class']] = dataset[[class_column_name]] # Add class column to dataset
    data.to_csv('dataset_processed.csv') # Save current dataframe to .csv
    print("----------------------------------------- Processed Dataset ------------------------------------------------------------")
    print(data)
    print("------------------------------------------------------------------------------------------------------------------------")
    print("\nFull Dataset Saved as 'dataset_processed.csv'...")
    
    check = input("Continue to Analysis (or do features need to be removed)? (y/n): ") # Check if dataframe is correctly formatted
    if check == 'y':
        feature_count = len(data.iloc[0,:])-1 # Number of columns minus class column
        break
    elif check == 'n':
        feat_remove = list(map(int, input("Which Features Do You Want to Remove?: ").split()))
        print("\nPreprocessing Data...\n")
        continue
    else:
        print("\n\n---INPUT ERROR---")
        sys.exit()

# Print information on dataset
print("\n----------- Data/Class Information  -----------")
print("Number of Features: ", feature_count)
classes = data.values[:,-1]
class_count = Counter(classes)
for key, value in class_count.items():
    per = value / len(classes) * 100
    print('Class=%s  Count=%d   Percentage=%.3f%%' % (key, value, per))
data = data.drop(columns=['class'])
print("-------------------------------------------------\n")

# Scaled data for some classification methods
scaler = StandardScaler()
data_std = scaler.fit_transform(data)


#------------------------------------------------- PLOT FEATURE DISTRIBUTIONS -------------------------------------------------#
# Plot each feature's distribution
print("Plotting Distribution of Each Feature...")
fig = plt.figure(figsize=(16,12))

for i, num in zip(data.columns, range(1,feature_count)):
    if i == 'class':
        continue
    else:
        i = int(i)
        ax = fig.add_subplot(4,4,num)
        data.hist(column="%d"%i, bins=50, ax=ax, range=[min(data["%d"%i]), max(data["%d"%i])], grid=False)
        ax.set_title("%d"%i, fontname="Times New Roman", fontsize=30)
        ax.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
        plt.setp(ax.spines.values(), color='black')
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(20)
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(20)
plt.tight_layout(pad=2)
print("Figure Saved as 'feature_distribution.png'...")
plt.savefig("feature_distribution.png")
#plt.savefig("feature_distribution.eps", format='eps')
plt.show()


#------------------------------------------------- CLASSIFICATION ALGORITHMS -------------------------------------------------#
# Kfold and Classification Scoring Function
def evaluate_model(X, y, model):
	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
	y_pred = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return y_pred

# List of Various Classification Algorithms
def models():
	models, names = list(), list()
    # Dummy
	models.append(DummyClassifier(strategy='most_frequent'))
	names.append('DUMMY')
    # LDA
	models.append(LinearDiscriminantAnalysis())
	names.append('LDA')
	# SVM
	models.append(SVC(kernel='linear'))
	names.append('SVM')
	# Bagging
	models.append(BaggingClassifier(n_estimators=1000))
	names.append('BAG')
	# RF
	models.append(RandomForestClassifier(n_estimators=1000))
	names.append('RF')
	# ET
	models.append(ExtraTreesClassifier(n_estimators=1000))
	names.append('ET')
	# XGB
	models.append(XGBClassifier())
	names.append('XGB')
	return models, names

# List of Various Classification Algorithms for SMOTE
def models_SMOTE():
	models, names = list(), list()
	# LR
	models.append(LogisticRegression(solver='lbfgs', multi_class='multinomial'))
	names.append('LR')
	# LDA
	models.append(LinearDiscriminantAnalysis())
	names.append('LDA')
	# SVM
	models.append(SVC(kernel='rbf'))
	names.append('SVM')
	# KNN
	models.append(KNeighborsClassifier(n_neighbors=3))
	names.append('KNN')
	return models, names

# Algorithm Testing
print("Beginning to Test Classification Method Accuracies...")
models, names = models()
results = list()
print("\n----------- Model Accuracy (STDEV) -----------")
for i in range(len(models)):
	if names[i] == 'SVM':
		y_pred = evaluate_model(data_std, classes, models[i])
		results.append(y_pred)
		print('>%s %.3f (%.3f)' % (names[i], mean(y_pred), std(y_pred)))
	else:
		y_pred = evaluate_model(data, classes, models[i])
		results.append(y_pred)
		print('>%s %.3f (%.3f)' % (names[i], mean(y_pred), std(y_pred)))
print("-----------------------------------------")

# Plot Boxplot 
plt.figure(figsize=(26,20))
plot = plt.boxplot(results, labels=names, showmeans=True, meanprops={'markersize':34})
for box in plot['boxes']:
    box.set(linewidth=6)
for flier in plot['fliers']:
    flier.set(linewidth=9)
for whisker in plot['whiskers']:
    whisker.set(linewidth=6)
for cap in plot['caps']:
    cap.set(linewidth=6)
for median in plot['medians']:
    median.set(linewidth=12, c='darkgreen')
plt.grid(False)
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
plt.xticks(fontfamily="Times New Roman", fontsize=100)
plt.yticks(fontfamily="Times New Roman", fontsize=100)
plt.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
plt.xlabel("Methods", fontsize=100, fontname="Times New Roman", labelpad=15)
plt.ylabel("Accuracy", fontsize=100, fontname="Times New Roman", labelpad=15)
plt.tight_layout()
print("Figure Saved as 'models_box_plot.png'...")
plt.savefig("models_box_plot.png")
#plt.savefig("models_box_plot.eps", format='eps')
plt.show()

# SMOTE Oversampling Testing
print("\nClassification Method Accuracies Using SMOTE...")
models, names = models_SMOTE()
results = list()
print("\n----------- SMOTE Model Accuracy (STDEV) -----------")
for i in range(len(models)):
	# Create pipeline
	steps = [('o', SMOTE(k_neighbors=2)), ('m', models[i])]
	pipeline = Pipeline(steps=steps)
	if names[i] == 'SVM':
		scores = evaluate_model(data_std, classes, pipeline)
		results.append(scores)
		print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
	else:
		scores = evaluate_model(data, classes, pipeline)
		results.append(scores)
		print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
print("------------------------------------------------------")

# Plot Boxplot 
plt.figure(figsize=(26,20))
plot = plt.boxplot(results, labels=names, showmeans=True, meanprops={'markersize':34})
for box in plot['boxes']:
    box.set(linewidth=6)
for flier in plot['fliers']:
    flier.set(linewidth=9)
for whisker in plot['whiskers']:
    whisker.set(linewidth=6)
for cap in plot['caps']:
    cap.set(linewidth=6)
for median in plot['medians']:
    median.set(linewidth=12, c='darkgreen')
plt.grid(False)
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
plt.xticks(fontfamily="Times New Roman", fontsize=100)
plt.yticks(fontfamily="Times New Roman", fontsize=100)
plt.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
plt.xlabel("Methods", fontsize=100, fontname="Times New Roman", labelpad=15)
plt.ylabel("Accuracy", fontsize=100, fontname="Times New Roman", labelpad=15)
plt.tight_layout()
print("Figure Saved as 'SMOTE_box_plot.png'...")
plt.savefig("SMOTE_box_plot.png")
#plt.savefig("SMOTE_box_plot.eps", format='eps')
plt.show()

#---------------------------------------------- CONFUSION MATRICES ----------------------------------------------#
# Makes KFold 
print("Separating Each Kfold...") 
y_test_list = []
y_pred_list = []
accuracy = []
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=1)

# Goes through list of classification methods of interest
for key,value in clf_dict.items():
    print("Starting Further Anaylsis for {}...".format(key))
    print("\n----- KFold Prediction with {} -----".format(key))
    count = 1
    if key == "SVM":
        data_split = pd.DataFrame(data_std)
    else:
        data_split = data
    for train_index, test_index in cv.split(data_split, classes):
        X_train, X_test, y_train, y_test = data_split.loc[train_index], data_split.loc[test_index], classes[train_index], classes[test_index]

        # Fitting classfication method on one kfold at a time
        value.fit(X_train, y_train)
        y_pred = value.predict(X_test)

        # Collects each test set and predictions
        for i in range(0, len(y_test)):
            y_test_list.append(y_test[i])
            y_pred_list.append(y_pred[i])
        
        print("KFold #{} with accuracy of {:.4f}".format(count, value.score(X_test, y_test)))
        count += 1
        accuracy.append(value.score(X_test, y_test))
    print("Average KFold Accuracy: {:.4f}".format(mean(accuracy)))
    print("--------------------------------------\n")

    # Output test and prediction lists
    kfold_pred = pd.DataFrame()
    kfold_pred[['TEST SET']] = pd.DataFrame(y_test_list)
    kfold_pred[['PREDICTION']] = pd.DataFrame(y_pred_list)
    print("Class Predictions Saved as '{}_kfold_predictions.csv'...".format(key))
    kfold_pred.to_csv("{}_kfold_predictions.csv".format(key))

    # Confusion matrix
    confusion = confusion_matrix(y_test_list, y_pred_list)
    print('\nCreating Confusion Matrix with {}...'.format(key))
    sum_list = []
    for i in range(0,len(class_count.keys())):
        for j in range(0,len(class_count.keys())):
            sum_list.append(sum(confusion[i]))
    
    info_list = []
    for i in range(0, len(confusion.flatten())):
        info_list.append(confusion.flatten()[i]/sum_list[i])

    # Make % labels
    pred_values = ["{0:0.0f}".format(value) for value in confusion.flatten()]
    info = ["{:.0%}".format(value) for value in info_list]
    labels = [f"{v1}\n({v2}, {v3})" for v1, v2, v3 in zip(pred_values,sum_list,info)]
    labels = np.asarray(labels).reshape(4,4)
    info_heatmap = np.asarray(info_list).reshape(4,4)
    df_cm = pd.DataFrame(info_heatmap, range(4), range(4))

    # Statistics
    print("\n---------------Classification Report for {}---------------".format(key))
    print(classification_report(y_test_list, y_pred_list, target_names=class_names))
    print("------------------------------------------------------------")
    
    #Plot Confusion Matrix
    plt.figure(figsize=(14,10))
    sns.set(font_scale=3)
    ax = sns.heatmap(df_cm, xticklabels=class_names, yticklabels=class_names, annot=labels, fmt='', 
            annot_kws={"size": 30, "fontfamily": "Times New Roman"}, linewidths=2, linecolor="black", cmap="vlag", cbar=False, vmin=0, vmax=1)
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    ax.axhline(y = 0, color = 'k', linewidth = 5)
    ax.axhline(y = 10, color = 'k',linewidth = 5)
    ax.axvline(x = 0, color = 'k',linewidth = 5)
    ax.axvline(x = 12, color = 'k',linewidth = 5)
    plt.title("Predicted Class", fontsize=60, fontname="Times New Roman", pad=20)
    plt.ylabel("True Class", fontsize=60, fontname="Times New Roman", labelpad=5)
    plt.xticks(fontfamily="Times New Roman", fontsize=35)
    plt.yticks(fontfamily="Times New Roman", fontsize=35)
    plt.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
    plt.tight_layout()
    print("\nFigure Saved as '{}_confusion_matrix.png'...\n".format(key))
    plt.savefig("{}_confusion_matrix.png".format(key))
    #plt.savefig("{}_confusion_matrix.eps".format(key), format='eps')
    plt.show()

#---------------------------------------------------- PR AND ROC CURVES ----------------------------------------------------#
#Add SVM and label classes
clf_dict["SVM"] = SVC(kernel='rbf', probability=True)
y = dataset[['class']]
Y = label_binarize(y, classes=[1, 2, 3, 4])
np.savetxt("class_binary.txt", Y)

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(data)

#One vs Rest
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=1)
classifiers = [SVC(kernel='rbf', probability=True), RandomForestClassifier(n_estimators=1000),  XGBClassifier()]
classifier_names = ['SVM (rbf)', 'RF', 'XGB']
class_names = ['Catastrophe', 'Growth', 'Rescue', 'Shortening']
colors = ['red', 'blue', 'green', 'orange']

# Get one k-fold for scaled data
Xstd_train_list = []
Xstd_test_list = []
ystd_train_list = []
ystd_test_list = []
X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []

# Currently set up to collect last kfold only
for train_index, test_index in cv.split(X, y):
    Xstd_train, Xstd_test, ystd_train, ystd_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
for train_index, test_index in cv.split(data, y):
    X_train, X_test, y_train, y_test = data.loc[train_index], data.loc[test_index], Y[train_index], Y[test_index]

# PR Curve
for state in range(4):
    print("\nStarting AUC Calculations for {}...\n".format(class_names[state]))
    plt.figure(figsize=(28,24))
    pr_aucs = {}
    for i in range(0, len(classifiers)):
        if i == 0:
            clf = OneVsRestClassifier(classifiers[i])
            clf.fit(Xstd_train, ystd_train)
            ystd_score = clf.predict_proba(Xstd_test)
            precision = dict()
            recall = dict()
            precision[i], recall[i], _ = precision_recall_curve(ystd_test[:, state], ystd_score[:, state])
            pr_auc = metrics.auc(recall[i], precision[i])
            plt.plot(recall[i], precision[i], lw=8, label=classifier_names[i], c=colors[i])
            pr_aucs["{}".format(list(clf_dict.keys())[i])] = pr_auc
            continue

        else:
            clf = OneVsRestClassifier(classifiers[i])
            clf.fit(X_train, y_train)
            y_score = clf.predict_proba(X_test)

            precision = dict()
            recall = dict()

            precision[i], recall[i], _ = precision_recall_curve(y_test[:, state], y_score[:, state])
            plt.plot(recall[i], precision[i], lw=8, label=classifier_names[i], c=colors[i])
            pr_auc = metrics.auc(recall[i], precision[i])
            pr_aucs["{}".format(list(clf_dict.keys())[i])] = pr_auc
            continue

    print("\n----------- AUC (PR) -----------")
    for key,value in pr_aucs.items():
        print(key, ":", value)
    print("--------------------------------------")

    font_size = 100
    plt.xlabel('Recall', fontname="Times New Roman", fontsize=font_size, labelpad=15)
    plt.ylabel('Precision', fontname="Times New Roman", fontsize=font_size, labelpad=15)
    plt.xticks(fontfamily="Times New Roman", fontsize=font_size)
    plt.yticks(fontfamily="Times New Roman", fontsize=font_size)
    plt.legend(loc="best", fontsize=80)
    plt.grid(False)
    #plt.title("Precision vs. Recall Curve for {}".format(class_names[state]), fontname="Times New Roman")
    plt.savefig("{}_PR_Curve.eps".format(class_names[state]), format='eps')
    plt.savefig("{}_PR_Curve.png".format(class_names[state]))
    plt.show()

# ROC Curve
    plt.figure(figsize=(28,24))
    roc_aucs = {}
    for i in range(0, len(classifiers)):
        fpr = dict()
        tpr = dict()

        if i == 0:
            fpr[i], tpr[i], _ = roc_curve(ystd_test[:, state], ystd_score[:, state])
            roc_auc = metrics.auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=8, label=classifier_names[i], c=colors[i])
            roc_aucs["{}".format(list(clf_dict.keys())[i])] = roc_auc
            continue

        else:
            clf = OneVsRestClassifier(classifiers[i])
            clf.fit(X_train, y_train)
            y_score = clf.predict_proba(X_test)
            fpr[i], tpr[i], _ = roc_curve(y_test[:, state], y_score[:, state])
            roc_auc = metrics.auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=8, label=classifier_names[i], c=colors[i])
            roc_aucs["{}".format(list(clf_dict.keys())[i])] = roc_auc
            continue

    print("\n----------- AUC (ROC) -----------")
    for key,value in pr_aucs.items():
        print(key, ":", value)
    print("--------------------------------------")

    plt.xlabel("False Positive Rate", fontname="Times New Roman", fontsize=font_size, labelpad=15)
    plt.ylabel("True Positive Rate", fontname="Times New Roman", fontsize=font_size, labelpad=15)
    plt.xticks(fontfamily="Times New Roman", fontsize=font_size)
    plt.yticks(fontfamily="Times New Roman", fontsize=font_size)
    plt.legend(loc="best", fontsize=80)
    plt.grid(False)
    #plt.title("ROC curve for {}".format(class_names[state]), fontname="Times New Roman")
    plt.savefig("{}_ROC_Curve.eps".format(class_names[state]), format='eps')
    plt.savefig("{}_ROC_Curve.png".format(class_names[state]))
    plt.show()

print("--------------")
print("End of Classification Analysis and Quality Testing")
