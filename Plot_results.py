import cv2 as cv
import numpy as np
from matplotlib import pylab
from sklearn.metrics import roc_curve
from itertools import cycle
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings("ignore")

def Statastical(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_convergence():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'WSO', 'WHO', 'GGO', 'SOA', 'PROPOSED']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv = np.zeros((Fitness.shape[-2], 5))
    for j in range(len(Algorithm) - 1):
        Conv[j, :] = Statastical(Fitness[j, :])
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv[j, :])
    print('-------------------------------------------------- Statistical Report of Dataset ', str(0 + 1),
          '  --------------------------------------------------')
    print(Table)

    length = np.arange(Fitness.shape[-1])
    Conv_Graph = Fitness

    plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, label='WSO')
    plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, label='WHO')
    plt.plot(length, Conv_Graph[2, :], color='#0cff0c', linewidth=3, label='GGO')
    plt.plot(length, Conv_Graph[3, :], color='#cf6275', linewidth=3, label='SOA')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, label='PROPOSED')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('Convergence Curve')
    plt.savefig("./Results/Convergence.png")
    plt.show()


def ROC_curve():
    lw = 2
    cls = ['RAN', 'GCNN', 'LSTM', 'SA-AMNet', 'PROPOSED']
    Actual = np.load('Targets.npy', allow_pickle=True).astype('int')
    colors = cycle(
        ["#fe2f4a", "#0165fc", "#ff028d", "lime", "black"])
    for i, color in zip(range(len(cls)), colors):
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i],
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


def Plot_Confusion():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    # cm = confusion_matrix(np.asarray(Actual), np.asarray(Predict))
    cm = confusion_matrix(np.asarray(Actual).argmax(axis=1), np.asarray(Predict).argmax(axis=1))
    Classes = ['5', '6', '7', '8']
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('Actual labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(Classes)
    ax.yaxis.set_ticklabels(Classes)
    path = "./Results/Confusion_Matrix.png"
    plt.title('Confusion matrix')
    plt.savefig(path)
    plt.show()


def Plot_Batchsize():
    eval = np.load('Eval_ALL_BS.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'lrplus', 'lrminus', 'DOR', 'prevalence']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17]

    for j in range(len(Graph_Terms)):
        Graph = eval[:, :, Graph_Terms[j] + 4]
        Graph = Graph[:4, :]
        X = np.arange(Graph.shape[0])
        ax = plt.axes()
        ax.set_facecolor("#FFEEF7")

        plt.plot(X, Graph[:, 0], color='#ff000d', linewidth=4, marker='$\spadesuit$', markerfacecolor='#ffff81',
                 markersize=12,
                 label="WSO")
        plt.plot(X, Graph[:, 1], color='#0cff0c', linewidth=4, marker='$\diamondsuit$', markerfacecolor='red',
                 markersize=12,
                 label="WHO")
        plt.plot(X, Graph[:, 2], color='#0652ff', linewidth=4, marker='$\clubsuit$', markerfacecolor='#bdf6fe',
                 markersize=12,
                 label="GGO")
        plt.plot(X, Graph[:, 3], color='#FFAE00', linewidth=4, marker='$\U0001F601$', markerfacecolor='yellow',
                 markersize=12,
                 label="SOA")
        plt.plot(X, Graph[:, 4], color='black', linewidth=4, marker='$\U00002660$', markerfacecolor='cyan',
                 markersize=12,
                 label="PROPOSED")
        plt.xticks(X, ('4', '16', '32', '64'))
        plt.grid(axis='y', linestyle='--', color='gray', which='major', alpha=0.8)
        plt.xlabel('Batch size', fontname="Franklin Gothic Medium", fontsize=12, fontweight='bold', color='#35530a')
        plt.ylabel(Terms[Graph_Terms[j]], fontname="Franklin Gothic Medium", fontsize=12, fontweight='bold',
                   color='#35530a')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False,
                   prop={'family': 'Comic Sans MS'})
        path = "./Results/Batch_size_%s_line.png" % (Terms[Graph_Terms[j]])
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Batch size vs ' + Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.1, 0.8, 0.8])
        X = np.arange(Graph.shape[0])
        ax.set_facecolor("#FFEEF7")

        ax.bar(X + 0.00, Graph[:, 5], color='#0652ff', width=0.15, label="RAN")
        ax.bar(X + 0.15, Graph[:, 6], color='#FF6600', width=0.15, label="GCNN")
        ax.bar(X + 0.30, Graph[:, 7], color='#02FF07', width=0.15, label="LSTM")
        ax.bar(X + 0.45, Graph[:, 8], color='#FF06E2', width=0.15, label="SA-AMNet")
        ax.bar(X + 0.60, Graph[:, 4], color='k', width=0.15, label="PROPOSED")
        plt.xticks(X + 0.3, ('4', '16', '32', '64'))
        plt.xlabel('Batch size', fontname="Franklin Gothic Medium", fontsize=12, color='#9d0759')
        plt.ylabel(Terms[Graph_Terms[j]], fontname="Franklin Gothic Medium", fontsize=12, color='#9d0759')
        plt.legend(loc='best', framealpha=0.5, fontsize=12, prop={'family': 'Comic Sans MS'})
        plt.grid(axis='y', linestyle='--', color='gray', which='major', alpha=0.8)
        path = "./Results/Batch_size_%s__bar.png" % (Terms[Graph_Terms[j]])
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Batch size vs ' + Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()


def Plot_Kfold():
    eval = np.load('Eval_ALL_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'lrplus', 'lrminus', 'DOR', 'prevalence']
    Table_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17]
    k_fold = ['1', '2', '3', '4', '5']
    Algorithm = ['TERMS', 'WSO', 'WHO', 'GGO', 'SOA', 'PROPOSED']
    Classifier = ['TERMS', 'RAN', 'GCNN', 'LSTM', 'SA-AMNet', 'PROPOSED']
    for k in range(eval.shape[1]):
        value = eval[k, :, 4:]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], (np.asarray(Terms))[np.asarray(Table_Term)])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, Table_Term])
        print('-------------------------------------------------- ', str(k_fold[k]), ' Fold of dataset', str(0 + 1),
              'Algorithm Comparison --------------------------------------------------')
        print(Table)
        Table = PrettyTable()
        Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Term)])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Table_Term])
        print('-------------------------------------------------- ', str(k_fold[k]), ' Fold of dataset', str(0 + 1),
              'Classifier Comparison --------------------------------------------------')
        print(Table)


def Sample_images():
    Audio = np.load('Audios.npy', allow_pickle=True)
    Images = np.load('Images.npy', allow_pickle=True)
    Text = np.load('Text.npy', allow_pickle=True)
    Target = np.load('Targets.npy', allow_pickle=True)
    if Target.shape[-1] >= 2:
        targ = np.argmax(Target, axis=1).reshape(-1, 1)
    else:
        targ = Target

    class_indices = {}
    for class_label in np.unique(targ):
        indices = np.where(targ == class_label)[0]
        class_indices[class_label] = indices
    for class_label, indices in class_indices.items():
        labels = ['5', '6', '7', '8']
        for i in range(5):
            print(labels[class_label], i + 1)

            Aud = Audio[indices[i]]
            Image = Images[indices[i]]
            txt = Text[indices[i]]

            plt.plot(Aud)
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude')
            plt.tight_layout()
            path = './Results/Sample_Images/Sample_' + str(labels[class_label]) + '_Aud_' + str(
                i + 1) + '.png'
            plt.savefig(path)
            plt.show()

            cv.imwrite('./Results/Sample_Images/Sample_' + str(labels[class_label]) + '_image_' + str(
                i + 1) + '.png', Image)

            print('Sample text : ' + str(labels[class_label]) + str(i + 1), txt)


if __name__ == '__main__':
    plot_convergence()
    ROC_curve()
    Plot_Confusion()
    Plot_Batchsize()
    Plot_Kfold()
    Sample_images()
