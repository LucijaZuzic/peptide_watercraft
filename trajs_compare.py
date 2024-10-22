import os
import pandas as pd
import scipy.stats as stats

sf1, sf2 = 5, 5

var_list = os.listdir("csv_results_traj_merged_validation/total/")
ws_long_list = os.listdir("csv_results_traj_merged_validation/old/" + var_list[0] + "/")

vehicle_zero = os.listdir("csv_results/1/1/")[0]
ride_zero = os.listdir("csv_results/1/1/" + vehicle_zero)[0]

dicti_wilcoxon_traj_new = {"variable": [], "ws": [], "model1": [], "model2": [], "u long": [], "p long": [], "u lat": [], "p lat": []}
dicti_wilcoxon_traj_merged = {"variable": [], "ws": [], "model1": [], "model2": [], "u long": [], "p long": [], "u lat": [], "p lat": []}

pd_file_old_wilcoxon = pd.read_csv("dicti_wilcoxon_traj_old.csv", index_col = False)

for ix in range(len(pd_file_old_wilcoxon["variable"])):
    dicti_wilcoxon_traj_merged["variable"].append(pd_file_old_wilcoxon["variable"][ix])
    dicti_wilcoxon_traj_merged["ws"].append(pd_file_old_wilcoxon["ws"][ix])
    dicti_wilcoxon_traj_merged["model1"].append(pd_file_old_wilcoxon["model1"][ix])
    dicti_wilcoxon_traj_merged["model2"].append(pd_file_old_wilcoxon["model2"][ix])
    dicti_wilcoxon_traj_merged["u long"].append(pd_file_old_wilcoxon["u long"][ix])
    dicti_wilcoxon_traj_merged["p long"].append(pd_file_old_wilcoxon["p long"][ix])
    dicti_wilcoxon_traj_merged["u lat"].append(pd_file_old_wilcoxon["u lat"][ix])
    dicti_wilcoxon_traj_merged["p lat"].append(pd_file_old_wilcoxon["p lat"][ix])

for var in var_list:
    for ws_long in ws_long_list:
        path_to_file = "csv_results_traj_merged_validation/total/" + var + "/" + ws_long
        pd_file = pd.read_csv(path_to_file, index_col = False)
        path_to_file_old = "csv_results_traj_merged_validation/old/" + var + "/" + ws_long
        pd_file_old = pd.read_csv(path_to_file_old, index_col = False)
        model_list_new = ["Bi", "Conv"]
        model_list = ["Bi", "Conv"]
        for col in pd_file_old:
            nc = col.replace(" long", "").replace(" lat", "")
            if nc not in model_list:
                model_list.append(nc)
        for model1_ix in [0, 1]:
            for model2_ix in range(model1_ix + 1, len(model_list)):
                try:
                    valu1 = pd_file[model_list[model1_ix] + " long"]
                    if model_list[model2_ix] + " long" in pd_file:
                        valu2 = pd_file[model_list[model2_ix] + " long"]
                    else:
                        valu2 = pd_file_old[model_list[model2_ix] + " long"]
                    uval1, pval1 = stats.wilcoxon(valu1[:min(len(valu1), len(valu2))], valu2[:min(len(valu1), len(valu2))])
                except:
                    uval1, pval1 = 1.0, 1.0
                try:
                    valu1 = pd_file[model_list[model1_ix] + " lat"]
                    if model_list[model2_ix] + " lat" in pd_file:
                        valu2 = pd_file[model_list[model2_ix] + " lat"]
                    else:
                        valu2 = pd_file_old[model_list[model2_ix] + " lat"]
                    uval2, pval2 = stats.wilcoxon(valu1[:min(len(valu1), len(valu2))], valu2[:min(len(valu1), len(valu2))])
                except:
                    uval2, pval2 = 1.0, 1.0
                print(model_list[model1_ix], model_list[model2_ix], uval1, pval1, uval2, pval2)
                dicti_wilcoxon_traj_new["variable"].append(var)
                dicti_wilcoxon_traj_new["ws"].append(ws_long.split("_")[0])
                dicti_wilcoxon_traj_new["model1"].append(model_list[model1_ix])
                dicti_wilcoxon_traj_new["model2"].append(model_list[model2_ix])
                dicti_wilcoxon_traj_new["u long"].append(uval1)
                dicti_wilcoxon_traj_new["p long"].append(pval1)
                dicti_wilcoxon_traj_new["u lat"].append(uval2)
                dicti_wilcoxon_traj_new["p lat"].append(pval2)

                dicti_wilcoxon_traj_merged["variable"].append(var)
                dicti_wilcoxon_traj_merged["ws"].append(ws_long.split("_")[0])
                dicti_wilcoxon_traj_merged["model1"].append(model_list[model1_ix])
                dicti_wilcoxon_traj_merged["model2"].append(model_list[model2_ix])
                dicti_wilcoxon_traj_merged["u long"].append(uval1)
                dicti_wilcoxon_traj_merged["p long"].append(pval1)
                dicti_wilcoxon_traj_merged["u lat"].append(uval2)
                dicti_wilcoxon_traj_merged["p lat"].append(pval2)

df_write = pd.DataFrame(dicti_wilcoxon_traj_new)
df_write.to_csv("dicti_wilcoxon_traj_new.csv", index = False)

df_dicti_wilcoxon_traj_merged = pd.DataFrame(dicti_wilcoxon_traj_merged)
df_dicti_wilcoxon_traj_merged.to_csv("dicti_wilcoxon_traj_merged.csv", index = False)