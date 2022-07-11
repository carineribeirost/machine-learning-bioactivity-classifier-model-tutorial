import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import *

from sklearn.model_selection import cross_val_score


#True and False Positive and Negative Metrics

def plot_cm(y_real, y_pred):
    """
    Plots a confusion matrix for given classifier predition data
    """

    cm = confusion_matrix(y_real, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    
# Sensitivity, hit rate, recall, or true positive rate
def tpr(y_real, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_real, y_pred).ravel()
    return TP/(TP + FN)


# Specificity or true negative rate
def tnr(y_real, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_real, y_pred).ravel()
    return TN/(TN + FP)

# Precision or positive predictive value
def ppv(y_real, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_real, y_pred).ravel()
    return TP/(TP + FP)

# Negative predictive value
def npv(y_real, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_real, y_pred).ravel()
    return TN/(TN + FN)

# Fall out or false positive rate
def fpr(y_real, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_real, y_pred).ravel()
    return FP/(FP + TN)

# False negative rate
def fnr(y_real, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_real, y_pred).ravel()
    return FN/(TP + FN)

# False discovery rate
def fdr(y_real, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_real, y_pred).ravel()
    return FP/(TP + FN)

#geometric mean
def G_mean(y_real, y_pred):
    return tpr(y_real, y_pred) * tnr(y_real, y_pred)

#Accuracy
def overall_accuracy(y_real, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_real, y_pred).ravel()
    return (TP+TN)/(TP+FP+FN+TN)
    
#dictionary with true and false postivie and negative parameters
pos_neg_rate = {
    'True_Positive_Rate' : tpr,
    'True_Negative_Rate' : tnr,
    'Positive_predictive_value' : ppv,
    'Negative_Predictive_Value' : npv,
    'False_Positive_Rate' : fpr,
    'False_Negative_Rate' : fnr,
    'False_Discovery_Rate' : fdr,
    'Overall_Accuracy': overall_accuracy,
                }
classifier_param = {
    'accuracy' : accuracy_score,
    'balanced_accuracy' : balanced_accuracy_score,
    'average_precision' : average_precision_score,
    'neg_brier_score' : brier_score_loss,
    'f1' : f1_score,
    'precision' : precision_score,
    'recall' : recall_score,
    'jaccard' : jaccard_score,
    'roc_auc' : roc_auc_score ,
    'G-mean' : G_mean,
    'MCC' : matthews_corrcoef
}


def classifier_scoring(y_real, y_pred, params = classifier_param):
    lista_classifier = list()
    for i in classifier_param:
        lista_classifier.append([i, classifier_param[i](y_real, y_pred)])
    df = pd.DataFrame(lista_classifier, columns = ['parameter', 'value'])
    return df

def pn_rate_df(y_real, y_pred):
    list_pos_neg = list()
    list_values_pn = list()
    for i in pos_neg_rate.values():
        list_pos_neg.append(i(y_real, y_pred))

    for i in pos_neg_rate.keys():
        list_values_pn.append(i)    
    return pd.DataFrame(list(zip(list_values_pn, list_pos_neg)),columns = ['parameter', 'value'])


sensitivity = make_scorer(recall_score, pos_label=1)

specificity = make_scorer(recall_score, pos_label=0)

mcc = make_scorer(matthews_corrcoef)

scoring = ['accuracy', 'average_precision','neg_brier_score',
           'f1', 'neg_log_loss', 'precision', sensitivity, specificity, mcc ,
           'jaccard','roc_auc']  

scoring_ = ['accuracy', 'average_precision','neg_brier_score',
           'f1', 'neg_log_loss', 'precision', 'sensitivity', 'specificity', 'MCC' ,
           'jaccard','roc_auc'] 
           
def k_fold_cv_evaluation(model_list, internal_x, internal_y, scores = scoring, scores_name = scoring_, k = 5):
    parameters_ = list()
    for j in scores:
        score_list = list()
        for i in model_list:
            score = cross_val_score(i, internal_x, internal_y, scoring = j, cv = k)
            score_list.append(np.mean(score))
        parameters_.append(np.mean(score_list))
    return pd.DataFrame(list(zip(scoring_, parameters_)),columns = ['parameter', 'value'])