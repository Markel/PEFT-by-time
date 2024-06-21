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
        "LoRA": 3213819904,
        "Prefix Tuning": 2689531904,
        "FT": 4218355712,
        "Parallel Adapters": 2951675904
    },
    "ag_news": {
        "LoRA": 6864961536,
        "Prefix Tuning": 3270443008,
        "FT": 8513323008,
        "Parallel Adapters": 5946408960
    },
    "commonsense_qa": {
        "LoRA": 2871984128,
        "Prefix Tuning": 2762932224,
        "FT": 3918462976,
        "Parallel Adapters": 2727280640
    },
    "race": {
        "LoRA": 11969429504,
        "Prefix Tuning": 6682509312,
        "FT": 17108566016,
        "Parallel Adapters": 10721624064
    }
})

# Divide every value by 1e-9 to get GB
df = df / 1e9

## INDIVIDUAL PLOT
sns.barplot(data=df[choosen], palette=pallete, order=order)
plt.ylabel("VRAM usage (GB)")
plt.savefig(f"./images/usage/vram_absolute_{choosen}.svg")

## COMBINED PLOT (absolute numbers)
f, axs = plt.subplots(ceil(len(datasets) / 2), 2, figsize=(10, 4 * ceil(len(datasets) / 2)),
                      sharex=True, sharey=True)

for i in range(len(datasets)):
    sns.barplot(data=df[datasets[i]], palette=pallete, order=order, ax=axs[i//2][i%2])
    axs[i//2][i%2].set_ylabel("VRAM usage (GB)")
    axs[i//2][i%2].set_title(datasets[i])
    axs[i//2][i%2].tick_params(axis='x', rotation=10)

plt.savefig(f"./images/usage/vram_absolute_all.svg")

## COMBINED PLOT FOR ALL (relative numbers)

relative_df = df.copy()

for dataset in datasets:
    ft_value = relative_df[dataset]["FT"]
    relative_df[dataset] = relative_df[dataset] / ft_value * 100

f, axs = plt.subplots(ceil(len(datasets) / 2), 2, figsize=(10, 4 * ceil(len(datasets) / 2)),
                      sharex=True, sharey=True)

for i in range(len(datasets)):
    sns.barplot(data=relative_df[datasets[i]], palette=pallete, order=order, ax=axs[i//2][i%2])
    axs[i//2][i%2].set_ylabel("Relative VRAM usage to FT (%)")
    axs[i//2][i%2].set_title(datasets[i])
    axs[i//2][i%2].tick_params(axis='x', rotation=10)

plt.savefig(f"./images/usage/vram_relative_all.svg")