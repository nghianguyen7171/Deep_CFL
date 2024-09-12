from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score
import numpy as np
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, classification_report, f1_score, roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from plot_metric.functions import BinaryClassification

def get_predictions(model, test_data):
    y_pred = model.predict(test_data)
    y_label = np.argmax(y_pred, axis=1)

    return  y_pred, y_label
# y_pred, y_label = get_predictions(dnn_clf, test_data_rrt)

def evaluation(y_test, y_label, y_pred):
    print('Accuracy: ', accuracy_score(y_test, y_label))
    print('Precision: ', precision_score(y_test, y_label, average='weighted'))
    print('Recall: ', recall_score(y_test, y_label, average='weighted'))
    print('F1 score: ', f1_score(y_test, y_label, average='weighted'))

    print('AUC: ', roc_auc_score(y_test, y_pred))
    print('Average precision score: ', average_precision_score(y_test, y_pred))

# evaluation(test_target_rrt, y_label, y_pred)
    
def calculate_ppv(y_true, y_pred_prob, threshold=0.5):
    """
    Calculate the Positive Predictive Value (PPV) or precision.

    Parameters:
        y_true (array-like): True labels.
        y_pred_prob (array-like): Predicted probabilities for the positive class.
        threshold (float): Threshold for converting probabilities to binary predictions (default is 0.5).

    Returns:
        float: The calculated Positive Predictive Value (PPV) or precision.
    """
    # Convert probabilities to binary predictions based on the threshold
    y_pred_binary = (y_pred_prob > threshold).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    # Calculate PPV (precision)
    ppv = tp / (tp + fp)

    return ppv

def calculate_sensitivity(y_true, y_pred_prob, threshold=0.5):
    """
    Calculate the Sensitivity or True Positive Rate (TPR).

    Parameters:
        y_true (array-like): True labels.
        y_pred_prob (array-like): Predicted probabilities for the positive class.
        threshold (float): Threshold for converting probabilities to binary predictions (default is 0.5).

    Returns:
        float: The calculated Sensitivity or True Positive Rate (TPR).
    """
    # Convert probabilities to binary predictions based on the threshold
    y_pred_binary = (y_pred_prob > threshold).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    # Calculate Sensitivity (TPR)
    sensitivity = tp / (tp + fn)

    return sensitivity

def calculate_specificity(y_true, y_pred_prob, threshold=0.5):
    """
    Calculate the Specificity or True Negative Rate (TNR).

    Parameters:
        y_true (array-like): True labels.
        y_pred_prob (array-like): Predicted probabilities for the positive class.
        threshold (float): Threshold for converting probabilities to binary predictions (default is 0.5).

    Returns:
        float: The calculated Specificity or True Negative Rate (TNR).
    """
    # Convert probabilities to binary predictions based on the threshold
    y_pred_binary = (y_pred_prob > threshold).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    # Calculate Specificity (TNR)
    specificity = tn / (tn + fp)

    return specificity



def bnr_rp(y_test, y_prob, y_label, labels):
    bc =  BinaryClassification(y_test, y_prob, labels=labels)

    # plots
    plt.figure(figsize=(15,10))
    plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
    bc.plot_roc_curve()
    plt.subplot2grid((2,6), (0,3), colspan=2)
    bc.plot_precision_recall_curve()
    plt.show()

    def F(beta, precision, recall):
    
        """
        Function that calculate f1, f2, and f0.5 scores.
        
        @params: beta, Float, type of f score
                precision: Float, average precision
                recall: Float, average recall
        
        @return: Float, f scores
        """
        
        return (beta*beta + 1)*precision*recall / (beta*beta*precision + recall)
    
    # precision, recall, and f1 f2 scores
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    print('f1 score {0:.4f}:'.format(F(1, np.mean(precision), np.mean(recall))))
    print('f2 score {0:.4f}:'.format(F(2, np.mean(precision), np.mean(recall))))
    print('precision {0:.4f}:'.format(precision_score(y_test, y_label, average='weighted')))
    print('recall {0:.4f}:'.format(recall_score(y_test, y_label, average='weighted')))
    print('AUPRC {0:.4f}:'.format(auc(recall, precision)))
    print('AUROC {0:.4f}:'.format(auc(fpr, tpr)))
    print('Acc {0:.4f}:'.format(accuracy_score(y_test, y_label)))

    # report
    bc.print_report()