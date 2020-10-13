# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:49:47 2018

@author: nelson.lin
"""

import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_curve,auc

def plot_cm(y_true,y_pred=None,normalize=True,figsize=(10,10),cmap=plt.cm.Blues):


    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    title='Confusion matrix'

    classes = list(set(y_true))

    cm = confusion_matrix(y_true = y_true,y_pred = y_pred,labels = classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    plt.show()

def auc_evaluation(y_true,y_score,pos_label=1,gradient = 0.3,delta_len = 100):
    fpr,tpr,thresholds = roc_curve(y_true,y_score,pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label = 'ROC Curve')

    plt.xlabel('FPR')

    plt.ylabel('TPR(Recall)')

    plt.title('ROC Graph:{:.4f}'.format(roc_auc),fontsize = 15)

    delta_len = delta_len

    delta_Xs = []
    for index in range(len(fpr)-delta_len):
        x_1 = fpr[index]
        x_2 = fpr[index+delta_len]
        delta_x = x_2-x_1
        delta_Xs.append(delta_x)

    delta_Ys = []
    for index in range(len(tpr)-delta_len):
        y_1 = tpr[index]
        y_2 = tpr[index+delta_len]
        delta_y = y_2-y_1
        delta_Ys.append(delta_y)

    gradients = np.array([np.round(delta_y/delta_x,2) for (delta_x, delta_y) in zip(delta_Xs,delta_Ys)])

    close_zero = np.argwhere(gradients ==gradient)[len(np.argwhere(gradients ==gradient))//2][0]

    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.plot(fpr[close_zero],tpr[close_zero],'o',markersize = 10, label = 'Cutoff Point',fillstyle = 'none', c = 'r', mew = 2)
    plt.text(fpr[close_zero]-0.1, tpr[close_zero]-0.15,
             s = 'Recall: {0:.2f}'.format(tpr[close_zero])
             +',FPR: {0:.2f},\n'.format(fpr[close_zero])
             +'Cutoff: {0:.4f},'.format(thresholds[close_zero])
            +' Gradient: {}'.format(gradient))
    plt.legend(loc = 4)


def corr_show(df,abs_corr = False,figsize = (12,9)):
    temp = df.copy()
    corrmat = temp.corr()
    if abs_corr:
        corrmat = np.abs(corrmat)
    plt.subplots(figsize=figsize)
    sns.heatmap(corrmat, vmax=0.9, square=True,cmap='Reds')
    plt.show()

def show_missing(df):
    df_na = (df.isnull().sum() / len(df)) * 100
    df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :df_na})
    f, ax = plt.subplots(figsize=(10, 8))
    plt.xticks(rotation='90')
    sns.barplot(y=df_na.index, x=df_na)
    plt.ylabel('Features', fontsize=15)
    plt.xlabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)

def percentile(df,column):
    percentile = np.array(range(1,len(df[column])+1))/len(df[column])
    plt.plot(percentile,df[column].sort_values())

def show_features_importance(model,features,top_k =20):
    plt.figure(figsize=(10,8))
    pd.Series(model.feature_importances_, features).sort_values(ascending = False)[:top_k][::-1].plot(kind = 'barh', color = 'c', label = 'feature importance')
    plt.title('Feature Importance', fontsize = 16)
    # plt.ylabel(front_size =10,)
    plt.legend(loc = 'best', fontsize = 10)
    plt.show()
