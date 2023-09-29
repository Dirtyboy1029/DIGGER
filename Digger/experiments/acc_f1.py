# -*- coding: utf-8 -*- 
# @Time : 2023/9/28 22:29 
# @Author : DirtyBoy 
# @File : acc_f1.py
'''

0.4600651611093562
0.2551217577962802
0.17597198072176534
0.1284896339510116

'''

import numpy as np
def evaluate(x_prob, gt_labels, threshold=0.4600651611093562, name='test'):


    x_pred = (x_prob >= threshold).astype(np.int32)

    # metrics
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
    accuracy = accuracy_score(gt_labels, x_pred)
    b_accuracy = balanced_accuracy_score(gt_labels, x_pred)

    MSG = "The accuracy on the {} dataset is {:.5f}%"
    print(MSG.format(name, accuracy * 100))
    MSG = "The balanced accuracy on the {} dataset is {:.5f}%"
    print(MSG.format(name, b_accuracy * 100))
    is_single_class = False
    if np.all(gt_labels == 1.) or np.all(gt_labels == 0.):
        is_single_class = True
    if not is_single_class:
        tn, fp, fn, tp = confusion_matrix(gt_labels, x_pred).ravel()

        fpr = fp / float(tn + fp)
        fnr = fn / float(tp + fn)
        f1 = f1_score(gt_labels, x_pred, average='binary')

        print("Other evaluation metrics we may need:")
        MSG = "False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
        print(MSG.format(fnr * 100, fpr * 100, f1 * 100))
        return MSG.format(fnr * 100, fpr * 100,
                          f1 * 100) + "The balanced accuracy on the {} dataset is {:.5f}%".format(name,
                                                                                                  accuracy * 100)