import pandas as pd
import numpy as np
import os

MAXINT = 10 ** 100

name_list_total = ["data_frame_val_merged"]

round_val = ["RMSE", "MAE", "MSE"]
round_val = ["RMSE"]
my_text_total = ""
my_appendix_total = ""

for name in name_list_total:
    df_dictio = pd.read_csv(name + ".csv", index_col = False)

    var_list = set(df_dictio["variable"])
    model_list  = set(df_dictio["model"])
    ws_list = set(df_dictio["ws"])

    min_max_for_metric_for_ws = dict()
    dictio = dict()
    use_stdev = False
    used_metric = set()
    for ix in range(len(df_dictio["variable"])):
        var = df_dictio["variable"][ix]
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
            used_metric.add(metric)
            if metric not in dictio[var][model][ws]:
                dictio[var][model][ws][metric] = []
            else:
                use_stdev = True
            dictio[var][model][ws][metric].append(df_dictio[metric][ix])

    dictio_stdev = dict()
    dictio_avg = dict()
    for var in dictio:
        for model in dictio[var]:
            for ws in dictio[var][model]:
                for metric in dictio[var][model][ws]:
                    if var not in dictio_avg:
                        dictio_avg[var] = dict()
                    if model not in dictio_avg[var]:
                        dictio_avg[var][model] = dict()
                    if ws not in dictio_avg[var][model]:
                        dictio_avg[var][model][ws] = dict()
                    if var not in dictio_stdev:
                        dictio_stdev[var] = dict()
                    if model not in dictio_stdev[var]:
                        dictio_stdev[var][model] = dict()
                    if ws not in dictio_stdev[var][model]:
                        dictio_stdev[var][model][ws] = dict()
                    if not use_stdev:
                        dictio_stdev[var][model][ws][metric] = 0
                        dictio_avg[var][model][ws][metric] = dictio[var][model][ws][metric][0]
                    else:
                        dictio_stdev[var][model][ws][metric] = np.std(dictio[var][model][ws][metric])
                        dictio_avg[var][model][ws][metric] = np.average(dictio[var][model][ws][metric])

                    if var not in min_max_for_metric_for_ws:
                        min_max_for_metric_for_ws[var] = dict()
                    if ws not in min_max_for_metric_for_ws[var]:
                        min_max_for_metric_for_ws[var][ws] = dict()
                    if metric not in min_max_for_metric_for_ws[var][ws]:
                        min_max_for_metric_for_ws[var][ws][metric] = (dictio_avg[var][model][ws][metric], model, dictio_avg[var][model][ws][metric], model)
                    else:    
                        metric_min, model_min, metric_max, model_max = min_max_for_metric_for_ws[var][ws][metric]
                        if dictio_avg[var][model][ws][metric] > metric_max:
                            metric_max = dictio_avg[var][model][ws][metric]
                            model_max = model
                        if dictio_avg[var][model][ws][metric] < metric_min:
                            metric_min = dictio_avg[var][model][ws][metric]
                            model_min = model
                        min_max_for_metric_for_ws[var][ws][metric] = (metric_min, model_min, metric_max, model_max)

    model_best_for = dict()
    for metric in round_val:
        if metric not in df_dictio:
            continue
        if "euclid" in metric:
            continue
        model_best_for[metric] = dict()
        for var in min_max_for_metric_for_ws:
            if "time" in var:
                continue
            model_best_for[metric][var] = dict()
            for model in dictio[var]:
                model_best_for[metric][var][model] = []
            for ws in min_max_for_metric_for_ws[var]:
                metric_min, model_min, metric_max, model_max = min_max_for_metric_for_ws[var][ws][metric]
                model_best = model_min
                if "R2" in metric:
                    model_best = model_max
                model_best_for[metric][var][model_best].append(ws)

    SOTA_vals = dict()
    SOTA_vals["speed"] = pd.read_csv("speed_SOTA.csv", index_col = False)
    SOTA_vals["longitude_no_abs"] = pd.read_csv("position_SOTA.csv", index_col = False)
    SOTA_vals["latitude_no_abs"] = SOTA_vals["longitude_no_abs"]
    conversion = {"speed": 3.6, "longitude_no_abs": 1 / (111 * 0.1 * 1000)}
    conversion["latitude_no_abs"] = conversion["longitude_no_abs"]
    if not "traj" in name:
        for metric in round_val:
            for var in SOTA_vals:
                print(var)
                usable_models = set()
                for model in model_best_for[metric][var]:
                    for ws in [2, 3, 4, 10]:
                        if ws in model_best_for[metric][var][model]:
                            usable_models.add(model)
                best_for_ws_one = dict()
                for ws in [2, 3, 4, 10]:
                    best_for_ws_one[ws] = ("", 1000000)
                for ws in [2, 3, 4, 10]:
                    for model_ix in range(len(SOTA_vals[var]["Model"])):
                        model_name = SOTA_vals[var]["Model"][model_ix]
                        model_val = SOTA_vals[var][str(ws)][model_ix]
                        if model_val < best_for_ws_one[ws][1]:
                            best_for_ws_one[ws] = (model_name, model_val)
                    for model_name in model_best_for[metric][var]:
                        model_val = dictio_avg[var][model_name][ws][metric]
                        if metric == "RMSE":
                            model_val = model_val / conversion[var]
                        if metric == "MSE":
                            model_val = np.sqrt(model_val) / conversion[var]
                        if metric == "MAE":
                            model_val = model_val / conversion[var]
                        if model_val < best_for_ws_one[ws][1]:
                            best_for_ws_one[ws] = (model_name, model_val)
                for model_ix in range(len(SOTA_vals[var]["Model"])):
                    str_row = SOTA_vals[var]["Model"][model_ix]
                    for ws in [2, 3, 4, 10]:
                        valu = SOTA_vals[var][str(ws)][model_ix]
                        if best_for_ws_one[ws][0] == SOTA_vals[var]["Model"][model_ix]:
                            str_row += " & $\\mathbf{" + str(valu) + "}$"
                        else:
                            str_row += " & $" + str(valu) + "$"
                    print(str_row + " \\\\ \\hline")
                for model in sorted(list(usable_models)):
                    str_row = model.replace("_", " ")
                    for ws in [2, 3, 4, 10]:
                        valu = dictio_avg[var][model][ws][metric]
                        if metric == "RMSE":
                            valu = valu / conversion[var]
                        if metric == "MSE":
                            valu = np.sqrt(valu) / conversion[var]
                        if metric == "MAE":
                            valu = valu / conversion[var]
                        if best_for_ws_one[ws][0] == model:
                            str_row += " & $\\mathbf{" + str(np.round(valu, 2)) + "}$"
                        else:
                            str_row += " & $" + str(np.round(valu, 2)) + "$"
                    print(str_row + " \\\\ \\hline")
                print(" ")