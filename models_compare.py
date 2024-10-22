import os
import pandas as pd
import scipy.stats as stats

sf1, sf2 = 5, 5

var_list = os.listdir("csv_results_merged_validation/total/")
ws_long_list = os.listdir("csv_results_merged_validation/old/" + var_list[0] + "/")

vehicle_zero = os.listdir("csv_results/1/1/")[0]
ride_zero = os.listdir("csv_results/1/1/" + vehicle_zero)[0]

dicti_wilcoxon_new = {"variable": [], "ws": [], "model1": [], "model2": [], "u": [], "p": []}
dicti_wilcoxon_merged = {"variable": [], "ws": [], "model1": [], "model2": [], "u": [], "p": []}

pd_file_old_wilcoxon = pd.read_csv("dicti_wilcoxon_old.csv", index_col = False)

for ix in range(len(pd_file_old_wilcoxon["variable"])):
    dicti_wilcoxon_merged["variable"].append(pd_file_old_wilcoxon["variable"][ix])
    dicti_wilcoxon_merged["ws"].append(pd_file_old_wilcoxon["ws"][ix])
    dicti_wilcoxon_merged["model1"].append(pd_file_old_wilcoxon["model1"][ix])
    dicti_wilcoxon_merged["model2"].append(pd_file_old_wilcoxon["model2"][ix])
    dicti_wilcoxon_merged["u"].append(pd_file_old_wilcoxon["u"][ix])
    dicti_wilcoxon_merged["p"].append(pd_file_old_wilcoxon["p"][ix])

for var in var_list:
    for ws_long in ws_long_list:
        path_to_file = "csv_results_merged_validation/total/" + var + "/" + ws_long
        pd_file = pd.read_csv(path_to_file, index_col = False)
        path_to_file_old = "csv_results_merged_validation/old/" + var + "/" + ws_long
        pd_file_old = pd.read_csv(path_to_file_old, index_col = False)
        model_list_new = ["Bi", "Conv"]
        model_list = ["Bi", "Conv"]
        for col in pd_file_old:
            model_list.append(col)
        for model1_ix in [0, 1]:
            for model2_ix in range(model1_ix + 1, len(model_list)):
                try:
                    valu1 = pd_file[model_list[model1_ix]]
                    if model_list[model2_ix] in pd_file:
                        valu2 = pd_file[model_list[model2_ix]]
                    else:
                        valu2 = pd_file_old[model_list[model2_ix]]
                    uval, pval = stats.wilcoxon(valu1, valu2)
                except:
                    uval, pval = 1.0, 1.0
                print(model_list[model1_ix], model_list[model2_ix], uval, pval)
                dicti_wilcoxon_new["variable"].append(var)
                dicti_wilcoxon_new["ws"].append(ws_long.split("_")[0])
                dicti_wilcoxon_new["model1"].append(model_list[model1_ix])
                dicti_wilcoxon_new["model2"].append(model_list[model2_ix])
                dicti_wilcoxon_new["u"].append(uval)
                dicti_wilcoxon_new["p"].append(pval)

                dicti_wilcoxon_merged["variable"].append(var)
                dicti_wilcoxon_merged["ws"].append(ws_long.split("_")[0])
                dicti_wilcoxon_merged["model1"].append(model_list[model1_ix])
                dicti_wilcoxon_merged["model2"].append(model_list[model2_ix])
                dicti_wilcoxon_merged["u"].append(uval)
                dicti_wilcoxon_merged["p"].append(pval)

df_write = pd.DataFrame(dicti_wilcoxon_new)
df_write.to_csv("dicti_wilcoxon_new.csv", index = False)

df_dicti_wilcoxon_merged = pd.DataFrame(dicti_wilcoxon_merged)
df_dicti_wilcoxon_merged.to_csv("dicti_wilcoxon_merged.csv", index = False)