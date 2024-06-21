# pylint: skip-file
import pandas as pd
from os import path
folder = path.join(path.dirname(__file__), "data")


tests = ["tweet_eval", "ag_news", "commonsense_qa", "race bleu", "race rogue", "LARGE_tweet_eval", "LARGE_commonsense_qa"]
metric = ["MulticlassAccuracy", "MulticlassAccuracy", "BinaryAccuracy", "BLEU4", "rougeL_fmeasure", "MulticlassAccuracy", "BinaryAccuracy"]
tests_plus_metric = zip(tests, metric)
techniques = ["FT", "LoRA", "Prefix Tuning", "Parallel Adapters"]
files = [
    ["FT_tweet_eval.json", "LoRA_tweet_eval.json", "prefix_tweet_eval_nt100.json", "adapters_tweet_eval.json"],
    ["FT_ag_news.json", "LoRA_ag_news.json", "prefix_ag_news_nt100.json", "adapters_ag_news_v2.json"],
    ["FT_commonsense_v2.json", "LoRA_commonsense_v2.json", "prefix_commonsense_v2_nt100.json", "adapters_commonsense_v2.json"],
    ["FT_race_b6.json", "LoRA_race.json", "prefix_race_nt100.json", "adapters_race.json"],
    ["FT_race_b6.json", "LoRA_race.json", "prefix_race_nt100.json", "adapters_race.json"],
    ["LARGE_FT_tweet_eval.json", "LARGE_LoRA_tweet_eval.json", "LARGE_prefix_tweet_eval_nt100.json", "LARGE_adapters_tweet_eval.json"],
    ["LARGE_FT_commonsense_v2.json", "LARGE_LoRA_commonsense_v2.json", "LARGE_prefix_commonsense_v2_nt100.json", "LARGE_adapters_commonsense_v2.json"]
]
unit = "GMACs" # steps_done or GMACs

checks = ["dev", "test"]
long_checks = []
for t in techniques:
    for c in checks:
        long_checks.append(c + " " + t)


df = pd.DataFrame(index=tests, columns=long_checks)



for i_test, (test, metric) in enumerate(tests_plus_metric):
    for i_tech, tech in enumerate(techniques):
        for check in checks:
            with open(path.join(folder, files[i_test][i_tech]), "r") as f:
                real_metric = check + "_" + metric
                data = pd.read_json(f)
                # Sort by metric
                data = data.sort_values(by=[real_metric, unit], ascending=[False, True])
                result = int(data[unit].iloc[0])
                if unit == "GMACs":
                    result = '{:.0e}'.format(float(result))
                if data[real_metric].iloc[0] in [0, 1]:
                    result = str(result) + "*"
                df.loc[test, check + " " + tech] = result

print(df)
print(df.to_latex(escape=False))