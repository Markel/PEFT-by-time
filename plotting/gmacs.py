# pylint: skip-file
from math import ceil
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_palette("colorblind", 10)

order = ["FT", "LoRA", "Prefix Tuning", "Parallel Adapters"]
pallete = ["#0173b2", "#de8f05", "#cc78bc", "#029e73"]
datasets = ["tweet_eval", "ag_news", "commonsense_qa", "race"]
choosen = datasets[0]


df = pd.DataFrame({
    "tweet_eval": {
        "LoRA": 35496344844.4,
        "Prefix Tuning": 23536876418.7,
        "FT": 52658103968.38,
        "Parallel Adapters": 35246481171.42
    },
    "ag_news": {
        "LoRA": 100879501577.71,
        "Prefix Tuning": 52434769842.03,
        "FT": 146308419430.19,
        "Parallel Adapters": 99827491416.75
    },
    "commonsense_qa": {
        "LoRA": 27819621337.66,
        "Prefix Tuning": 21094504609.98,
        "FT": 41436716724.14,
        "Parallel Adapters": 27694858922.7
    },
    "race": {
        "LoRA": 178929673424.47,
        "Prefix Tuning": 122272266726.79,
        "FT": 257499383804.11,
        "Parallel Adapters": 178011675631.51
    }
})


# Divide every value by 1e-9 to get GB
df = df / 1e9

## INDIVIDUAL PLOT
sns.barplot(data=df[choosen], palette=pallete, order=order)
plt.ylabel("GMACs per step")
plt.savefig(f"./images/usage/gmacs_absolute_{choosen}.svg")

## COMBINED PLOT (absolute numbers)
f, axs = plt.subplots(ceil(len(datasets) / 2), 2, figsize=(10, 4 * ceil(len(datasets) / 2)),
                      sharex=True, sharey=True)

for i in range(len(datasets)):
    sns.barplot(data=df[datasets[i]], palette=pallete, order=order, ax=axs[i//2][i%2])
    axs[i//2][i%2].set_ylabel("GMACs per step")
    axs[i//2][i%2].set_title(datasets[i])
    axs[i//2][i%2].tick_params(axis='x', rotation=10)

plt.savefig(f"./images/usage/gmacs_absolute_all.svg")

## COMBINED PLOT FOR ALL (relative numbers)

relative_df = df.copy()

for dataset in datasets:
    ft_value = relative_df[dataset]["FT"]
    relative_df[dataset] = relative_df[dataset] / ft_value * 100

f, axs = plt.subplots(ceil(len(datasets) / 2), 2, figsize=(10, 4 * ceil(len(datasets) / 2)),
                      sharex=True, sharey=True)

for i in range(len(datasets)):
    sns.barplot(data=relative_df[datasets[i]], palette=pallete, order=order, ax=axs[i//2][i%2])
    axs[i//2][i%2].set_ylabel("Relative GMACs per step to FT (%)")
    axs[i//2][i%2].set_title(datasets[i])
    axs[i//2][i%2].tick_params(axis='x', rotation=10)

plt.savefig(f"./images/usage/gmacs_relative_all.svg")