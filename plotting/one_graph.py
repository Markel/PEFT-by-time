# pylint: skip-file
from cProfile import label
from os import path
from pdb import run
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set_palette("colorblind", 10)

folder = path.join(path.dirname(__file__), "data")



running_average = 1
order = ["FT", "LoRA", "Prefix Tuning", "Parallel Adapters"]
pallete = {"FT": "#0173b2", "LoRA": "#de8f05", "Prefix Tuning": "#cc78bc", "Parallel Adapters": "#029e73"}

y_axises = ['train_MulticlassAccuracy', 'train_MulticlassPrecision', 'train_MulticlassRecall', 'train_MulticlassF1Score', 'dev_MulticlassAccuracy', 'dev_MulticlassPrecision', 'dev_MulticlassRecall', 'dev_MulticlassF1Score', 'test_MulticlassAccuracy', 'test_MulticlassPrecision', 'test_MulticlassRecall', 'test_MulticlassF1Score', 'train_loss', 'dev_loss', 'test_loss']

y_labels = ["Multiclass Accuracy", "Multiclass Precision", "Multiclass Recall", "Multiclass F1 Score", "Multiclass Accuracy", "Multiclass Precision", "Multiclass Recall", "Multiclass F1 Score", "Multiclass Accuracy", "Multiclass Precision", "Multiclass Recall", "Multiclass F1 Score", "Loss", "Loss", "Loss"]

# name = "tweet_eval"
# files = ["FT_tweet_eval-c5z5t6t0.json", "LoRA_tweet_eval-i6bjh9g0.json",
#          "prefix_tweet_eval_nt100-tfjeb7cd.json", "adapters_tweet_eval-efszwg9n.json"]

name = "ag_news"
files = ["FT_ag_news.json", "LoRA_ag_news.json",
         "prefix_ag_news_nt100.json", "adapters_ag_news_v2.json"]

# name = "commonsense_qa"
# files = ["FT_commonsense_v2-sf9otr4l.json", "LoRA_commonsense_v2-jzzy5a4g.json",
#          "prefix_commonsense_v2_nt100-jev57kmm.json", "adapters_commonsense_v2-glufvsgb.json"]
# y_axises = ['train_BinaryAccuracy', 'train_BinaryPrecision', 'train_BinaryRecall', 'train_BinaryF1Score', 'dev_BinaryAccuracy', 'dev_BinaryPrecision', 'dev_BinaryRecall', 'dev_BinaryF1Score', 'test_BinaryAccuracy', 'test_BinaryPrecision', 'test_BinaryRecall', 'test_BinaryF1Score', 'train_loss', 'dev_loss', 'test_loss']
# y_labels = ["Binary Accuracy", "Binary Precision", "Binary Recall", "Binary F1 Score", "Binary Accuracy", "Binary Precision", "Binary Recall", "Binary F1 Score", "Binary Accuracy", "Binary Precision", "Binary Recall", "Binary F1 Score", "Loss", "Loss", "Loss"]

# name = "race"
# files = ["FT_race_b6-wg320qpe.json", "LoRA_race-mqe7bdqa.json",
#          "prefix_race_nt100-ng9up6u4.json", "adapters_race-x3klzyxd.json"]
# y_axises = ['train_BLEU1', 'train_BLEU2', 'train_BLEU3', 'train_BLEU4', 'train_rougeL_fmeasure', 'train_rougeL_precision', 'train_rougeL_recall', 'dev_BLEU1', 'dev_BLEU2', 'dev_BLEU3', 'dev_BLEU4', 'dev_rougeL_fmeasure', 'dev_rougeL_precision', 'dev_rougeL_recall', 'test_BLEU1', 'test_BLEU2', 'test_BLEU3', 'test_BLEU4', 'test_rougeL_fmeasure', 'test_rougeL_precision', 'test_rougeL_recall', 'train_loss', 'dev_loss', 'test_loss']
# y_labels = ["BLEU1", "BLEU2", "BLEU3", "BLEU4", "Rouge-L F1 Score", "Rouge-L Precision", "Rouge-L Recall", "BLEU1", "BLEU2", "BLEU3", "BLEU4", "Rouge-L F1 Score", "Rouge-L Precision", "Rouge-L Recall", "BLEU1", "BLEU2", "BLEU3", "BLEU4", "Rouge-L F1 Score", "Rouge-L Precision", "Rouge-L Recall", "Loss", "Loss", "Loss"]

# name = "VeRA_commonsense_qa"
# files = ["FT_commonsense_v2.json", "LoRA_commonsense_v2.json",
#          "VeRA_commonsense_noGMACs.json"]
# y_axises = ['train_BinaryAccuracy', 'train_BinaryPrecision', 'train_BinaryRecall', 'train_BinaryF1Score', 'dev_BinaryAccuracy', 'dev_BinaryPrecision', 'dev_BinaryRecall', 'dev_BinaryF1Score', 'test_BinaryAccuracy', 'test_BinaryPrecision', 'test_BinaryRecall', 'test_BinaryF1Score', 'train_loss', 'dev_loss', 'test_loss']
# y_labels = ["Binary Accuracy", "Binary Precision", "Binary Recall", "Binary F1 Score", "Binary Accuracy", "Binary Precision", "Binary Recall", "Binary F1 Score", "Binary Accuracy", "Binary Precision", "Binary Recall", "Binary F1 Score", "Loss", "Loss", "Loss"]
# order = ["FT", "LoRA", "VeRA"]
# pallete = {"FT": "#0173b2", "LoRA": "#de8f05", "VeRA": "#ca9161"}


x_lims = [[0, 30000], [0, 100000]]
x_lims_activated: bool = False

x_axis = ["steps_done", "GMACs"] # "GMACs" or "steps_done"

def moving_average(a, n=3):
    # If the first value is na it will be replaced by the second value
    if np.isnan(a[0]):
        a[0] = a[1]
    # Moving average
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


df = pd.DataFrame(index=order, columns=x_axis+y_axises)

for t in range(len(files)):
    with open(path.join(folder, files[t]), "r") as f:
        data = pd.read_json(f)
        # Get each column of the file, create a numpy array and add it in the dataframe
        for y in x_axis+y_axises:
            df.loc[order[t], y] = np.array(data[y])

# Apply the moving average to all the columns contained in y_axises and store them in the dataframe
if running_average > 1:
    for m in y_axises:
        for t in order:
            df.loc[t, m] = moving_average(df[m][t], running_average)

for x_metric, h_lims in list(zip(x_axis, x_lims)):
    for m, m_label in list(zip(y_axises, y_labels)):
        plt.figure(layout="constrained")
        for t in order:
            sns.lineplot(x=df[x_metric][t][:len(df[m][t])], y=df[m][t], color=pallete[t], markers=False, label=t)
        plt.ylabel(m_label)
        plt.xlabel(x_metric)
        if x_lims_activated: plt.xlim(h_lims)
        #plt.legend()
        plt.savefig(f"./images/results/single/{name}/{name}_{x_metric}_{m}_s{running_average}.svg")
        
        plt.close()
