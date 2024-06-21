# pylint: skip-file
import pandas as pd
from os import path
import seaborn as sns
import matplotlib.pyplot as plt
folder = path.join(path.dirname(__file__), "data")


tests = ["TweetEval-Hate", "AG News", "CommonsenseQA", "RACE"]
metric = ["MulticlassAccuracy", "MulticlassAccuracy", "BinaryAccuracy", "rougeL_fmeasure"]
tests_plus_metric = zip(tests, metric)
techniques = ["FT", "LoRA", "Prefix Tuning", "Parallel Adapters"]
pallete = {"FT": "#0173b2", "LoRA": "#de8f05", "Prefix Tuning": "#cc78bc", "Parallel Adapters": "#029e73"}
files = [
    ["FT_tweet_eval.json", "LoRA_tweet_eval.json", "prefix_tweet_eval_nt100.json", "adapters_tweet_eval.json"],
    ["FT_ag_news.json", "LoRA_ag_news.json", "prefix_ag_news_nt100.json", "adapters_ag_news_v2.json"],
    ["FT_commonsense_v2.json", "LoRA_commonsense_v2.json", "prefix_commonsense_v2_nt100.json", "adapters_commonsense_v2.json"],
    ["FT_race_b6.json", "LoRA_race.json", "prefix_race_nt100.json", "adapters_race.json"]
]
unit = "GMACs" # steps_done or GMACs

checks = ["test"]
long_checks = []
for t in techniques:
    for c in checks:
        long_checks.append(c + " " + t)


df = pd.DataFrame(columns=[unit, "value", "technique", "dataset"])



for i_test, (test, metric) in enumerate(tests_plus_metric):
    for i_tech, tech in enumerate(techniques):
        for check in checks:
            with open(path.join(folder, files[i_test][i_tech]), "r") as f:
                real_metric = check + "_" + metric
                data = pd.read_json(f)
                # Sort by metric
                data = data.sort_values(by=[real_metric, unit], ascending=[False, True])
                result = {unit: data[unit].iloc[0], "value": data[real_metric].iloc[0], "technique": tech, "dataset": test}
                # Add result as a new row in df not using append
                result = pd.DataFrame(result, index=[0])
                df = pd.concat([df, result], ignore_index=True)

print(df)

fig = plt.figure(figsize=(10, 4), layout='constrained')
ax = plt.subplot(111)
sns.scatterplot(data=df, x=unit, y="value", hue="technique", style="dataset", palette=pallete, markers=["o", "X", "s", "<"],  s=150, ax=ax)
#ax.set_xscale('log')
ax.set_ylabel('score')
handles, labels = ax.get_legend_handles_labels()
ax.get_legend().remove()
hanlab = list(zip(handles, labels))
nhandlab = []
assert len(hanlab) % 2 == 0
for i in range(len(hanlab)//2):
    nhandlab.append(hanlab[i])
    nhandlab.append(hanlab[i+len(hanlab)//2])
handles, labels = zip(*nhandlab)
fig.legend(handles, labels, loc='outside upper center', ncol=(len(handles)+1)//2)
plt.savefig(f"./images/speed_of_each_method_in_datasets_in_{unit}.svg")
plt.show()
