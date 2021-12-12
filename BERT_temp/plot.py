from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle

mpl.use('Agg')

def pr_curve(mode_list, figure_id):
    for mode in mode_list:
        true_path = "./data/{mode}_tmp_save/y_true.pkl".format(mode=mode['file'])
        scores_path = "./data/{mode}_tmp_save/y_scores.pkl".format(mode=mode['file'])
        if os.path.isfile(true_path) and os.path.isfile(scores_path):
            with open(true_path, 'rb') as pkl_file:
                y_true = pickle.load(pkl_file)
            with open(scores_path, 'rb') as pkl_file:
                y_scores = pickle.load(pkl_file)

        precisions, recalls, threshold = precision_recall_curve(y_true, y_scores)
        plt.plot(recalls, precisions, color=mode['color'], marker=mode['marker'], markevery=500, ms=6, lw=1.5, label=mode['label'])
        print("Finished ploting {name}".format(name=mode['label']))

    legend_font = {"weight": "black"}
    plt.ylim([0.6, 1.0])
    plt.xlim([0.0, 0.4])
    plt.xticks(fontweight="black")
    plt.yticks(fontweight="black")
    plt.legend(loc="upper right", prop=legend_font)
    plt.xlabel('Recall', fontweight="black")
    plt.ylabel('Precision', fontweight="black")
    plt.grid(True)
    dir_path = "./figure"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig('./figure/PRcurve{figure_id}.pdf'.format(figure_id=figure_id))
    plt.clf()
