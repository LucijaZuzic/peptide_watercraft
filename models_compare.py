import pandas as pd
import scipy.stats as stats

round_val = ["R2", "MAE", "RMSE", "MSE"]

df_dictio = pd.read_csv("data_frame_val_merged.csv", index_col = False)

dictio = dict()
for ix in range(len(df_dictio["variable"])):
    var = df_dictio["variable"][ix]
    if "time" in var:
        continue
    model = df_dictio["model"][ix]
    ws = df_dictio["ws"][ix]
    if var not in dictio:
        dictio[var] = dict()
    if model not in dictio[var]:
        dictio[var][model] = dict()
    if ws not in dictio[var][model]:
        dictio[var][model][ws] = dict()
    for metric in round_val:
        if metric not in df_dictio:
            continue
        if metric not in dictio[var][model][ws]:
            dictio[var][model][ws][metric] = []
        else:
            use_stdev = True
        dictio[var][model][ws][metric].append(df_dictio[metric][ix])

var_list = list(dictio.keys())
model_list = sorted(list(dictio[var_list[0]].keys()))
ws_list = list(dictio[var_list[0]][model_list[0]].keys())
metric_list = list(dictio[var_list[0]][model_list[0]][ws_list[0]].keys())
for test_var in ["wilcoxon", "mann_whitney"]:
    for metric in metric_list:
        if test_var == "wilcoxon":
            dicti_wilcoxon_new = {"variable": [], "ws": [], "model1": [], "model2": [], "u": [], "p": []}
            for var in var_list:
                for ws in ws_list:
                    for model1_ix in range(len(model_list)):
                        model1 = model_list[model1_ix]
                        for model2_ix in range(model1_ix + 1, len(model_list)):
                            model2 = model_list[model2_ix]
                            try:    
                                uval, pval = stats.wilcoxon(dictio[var][model1][ws][metric], dictio[var][model2][ws][metric])
                            except:
                                uval, pval = 1.0, 1.0
                            print(metric, var, ws, model1, model2, uval, pval)
                            dicti_wilcoxon_new["variable"].append(var)
                            dicti_wilcoxon_new["ws"].append(ws)
                            dicti_wilcoxon_new["model1"].append(model1)
                            dicti_wilcoxon_new["model2"].append(model2)
                            dicti_wilcoxon_new["u"].append(uval)
                            dicti_wilcoxon_new["p"].append(pval)
            df_write = pd.DataFrame(dicti_wilcoxon_new)
            df_write.to_csv("dicti_wilcoxon_" + metric + ".csv", index = False)
        if test_var == "mann_whitney":
            dicti_mann_whitney_new = {"variable": [], "ws": [], "model1": [], "model2": [], "u": [], "p": []}
            for var in var_list:
                for ws in ws_list:
                    for model1_ix in range(len(model_list)):
                        model1 = model_list[model1_ix]
                        for model2_ix in range(model1_ix + 1, len(model_list)):
                            model2 = model_list[model2_ix]
                            uval, pval = stats.mannwhitneyu(dictio[var][model1][ws][metric], dictio[var][model2][ws][metric])
                            print(metric, var, ws, model1, model2, uval, pval)
                            dicti_mann_whitney_new["variable"].append(var)
                            dicti_mann_whitney_new["ws"].append(ws)
                            dicti_mann_whitney_new["model1"].append(model1)
                            dicti_mann_whitney_new["model2"].append(model2)
                            dicti_mann_whitney_new["u"].append(uval)
                            dicti_mann_whitney_new["p"].append(pval)
            df_write = pd.DataFrame(dicti_mann_whitney_new)
            df_write.to_csv("dicti_mann_whitney_" + metric + ".csv", index = False)