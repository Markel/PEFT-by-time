# pylint: skip-file
from cProfile import label
from os import path
from pdb import run
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch import ne

sns.set_palette("colorblind", 10)

folder = path.join(path.dirname(__file__), "data")

running_average = 15
order = ["FT", "LoRA", "Prefix Tuning", "Parallel Adapters"]
pallete = {"FT": "#0173b2", "LoRA": "#de8f05", "Prefix Tuning": "#cc78bc", "Parallel Adapters": "#029e73"}

y_axises = ['MulticlassAccuracy', 'MulticlassPrecision', 'MulticlassRecall', 'MulticlassF1Score', 'loss']
y_labels = ["Multiclass Accuracy", "Multiclass Precision", "Multiclass Recall", "Multiclass F1 Score", "Loss"]

y_axises = ['MulticlassAccuracy', 'loss']
y_labels = ["Multiclass Accuracy", "Loss"]

# name = "tweet_eval"
# files = ["FT_tweet_eval.json", "LoRA_tweet_eval.json",
#          "prefix_tweet_eval_nt100.json", "adapters_tweet_eval.json"]

# name = "LARGE_tweet_eval"
# files = ["LARGE_FT_tweet_eval.json", "LARGE_LoRA_tweet_eval.json",
#          "LARGE_prefix_tweet_eval_nt100.json", "LARGE_adapters_tweet_eval.json"]

# name = "ag_news"
# files = ["FT_ag_news.json", "LoRA_ag_news.json",
#          "prefix_ag_news_nt100.json", "adapters_ag_news_v2.json"]

# name = "commonsense_qa"
# files = ["FT_commonsense_v2.json", "LoRA_commonsense_v2.json",
#          "prefix_commonsense_v2_nt100.json", "adapters_commonsense_v2.json"]

# y_axises = ['BinaryAccuracy', 'BinaryPrecision', 'BinaryRecall', 'BinaryF1Score', 'loss']
# y_labels = ["Binary Accuracy", "Binary Precision", "Binary Recall", "Binary F1 Score", "Loss"]


name = "check_commonsense_qa"
files = ["Check_FT_commonsense.json", "LoRA_commonsense_v2.json",]

y_axises = ['BinaryAccuracy']
y_labels = ["Binary Accuracy"]

order = ["FT", "LoRA"]
pallete = {"FT": "#0173b2", "LoRA": "#de8f05"}

# name = "LARGE_commonsense_qa"
# files = ["LARGE_FT_commonsense_v2.json", "LARGE_LoRA_commonsense_v2.json",
#          "LARGE_prefix_commonsense_v2_nt100.json", "LARGE_adapters_commonsense_v2.json"]

# y_axises = ['BinaryAccuracy', 'BinaryPrecision', 'BinaryRecall', 'BinaryF1Score', 'loss']
# y_labels = ["Binary Accuracy", "Binary Precision", "Binary Recall", "Binary F1 Score", "Loss"]

# name = "race"
# files = ["FT_race_b6.json", "LoRA_race.json",
#          "prefix_race_nt100.json", "adapters_race.json"]
# y_axises = ['BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'rougeL_fmeasure', 'rougeL_precision', 'rougeL_recall', 'loss']
# y_labels = ["BLEU1", "BLEU2", "BLEU3", "BLEU4", "Rouge-L F1 Score", "Rouge-L Precision", "Rouge-L Recall", "Loss"]

# # y_axises = ['BLEU1', 'BLEU2', 'BLEU3', 'BLEU4']
# # y_labels = ["BLEU1", "BLEU2", "BLEU3", 'BLEU4']
# y_axises = ['BLEU4', 'rougeL_fmeasure', 'loss']
# y_labels = ["BLEU4", "Rouge-L F1 Score", "Loss"]

# name = "VeRA_commonsense_qa"
# files = ["FT_commonsense_v2.json", "LoRA_commonsense_v2.json",
#          "VeRA_commonsense_noGMACs.json"]
# # y_axises = ['BinaryAccuracy', 'BinaryPrecision', ]
# y_axises = ['BinaryRecall', 'BinaryF1Score', 'loss']
# # y_labels = ["Binary Accuracy", "Binary Precision",]
# y_labels = [ "Binary Recall", "Binary F1 Score", "Loss"]
# order = ["FT", "LoRA", "VeRA"]
# pallete = {"FT": "#0173b2", "LoRA": "#de8f05", "VeRA": "#ca9161"}



x_lims = [[0, 30000], [0, 100000]]
x_lims_activated: bool = False

x_axis = ["steps_done", "GMACs"] # "GMACs" or "steps_done"


new_y_axis = []

for i in range(len(y_axises)):
    new_y_axis.append("train_" + y_axises[i])
    new_y_axis.append("dev_" + y_axises[i])
    new_y_axis.append("test_" + y_axises[i])

def moving_average(a, n=3):
    # If the first value is na it will be replaced by the second value
    if np.isnan(a[0]):
        a[0] = a[1]
    # Moving average
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


df = pd.DataFrame(index=order, columns=x_axis+new_y_axis)

for t in range(len(files)):
    with open(path.join(folder, files[t]), "r") as f:
        data = pd.read_json(f)
        # Get each column of the file, create a numpy array and add it in the dataframe
        for y in x_axis+new_y_axis:
            df.loc[order[t], y] = np.array(data[y])

# Apply the moving average to all the columns contained in y_axises and store them in the dataframe
if running_average > 1:
    for m in new_y_axis:
        for t in order:
            df.loc[t, m] = moving_average(df[m][t], running_average)

for x_metric, h_lims in list(zip(x_axis, x_lims)):
    # I now sharex could be True but readability
    f, axs = plt.subplots(len(y_labels), 3, figsize=(12, 4*len(y_labels)), sharex="row", sharey="row", layout='constrained')
    if len(y_labels) == 1:
        axs = [axs]
    sns.set_context("paper")
    all_m = list(zip(y_axises, y_labels))
    for i in range(len(y_axises)):
        m, m_label = all_m[i]
        #print(m_label)
        ax1, ax2, ax3 = axs[i]
        for t in order:
            sns.lineplot(x=df[x_metric][t][:len(df["train_" + m][t])], y=df["train_" + m][t],
                         color=pallete[t], markers=False, label=t, ax=ax1)
            sns.lineplot(x=df[x_metric][t][:len(df["dev_" + m][t])], y=df["dev_" + m][t],
                         color=pallete[t], markers=False, label=t, ax=ax2)
            sns.lineplot(x=df[x_metric][t][:len(df["test_" + m][t])], y=df["test_" + m][t],
                         color=pallete[t], markers=False, label=t, ax=ax3)
        ax1.set_ylabel(m_label)
        ax2.set_ylabel(m_label)
        ax3.set_ylabel(m_label)
        ax1.set_xlabel(x_metric)
        ax2.set_xlabel(x_metric)
        ax3.set_xlabel(x_metric)
        # Add dotted horizontal line at 0.2
        ax1.axhline(y=0.2, color='k', linestyle='--', label="Random")
        ax2.axhline(y=0.2, color='k', linestyle='--', label="Random")
        ax3.axhline(y=0.2, color='k', linestyle='--', label="Random")

        if i == 0:
            ax1.set_title("Train")
            ax2.set_title("Dev")
            ax3.set_title("Test")
        if x_lims_activated:
            ax1.xlim(h_lims)
            ax2.xlim(h_lims)
            ax3.xlim(h_lims)
        ax1.get_legend().remove()
        ax2.get_legend().remove()
        ax3.get_legend().remove()
        handles, labels = ax1.get_legend_handles_labels()
        f.legend(handles, labels, loc='outside lower center', ncol=len(labels))
    plt.savefig(f"./images/results/specific/{name}/{name}_{x_metric}_s{running_average}.svg")
    plt.close()
