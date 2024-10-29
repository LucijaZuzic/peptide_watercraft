import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

MAXINT = 10 ** 100
cm = 1/2.54  # centimeters in inches

def stringify(value_round, rounding, skip_mul):
    if value_round == 0:
        return "0", 0
    if abs(value_round) >= 1 or skip_mul:
        return str(np.round(value_round, rounding)), 0
    else:
        pot = 0
        while abs(value_round) < 1:
            pot += 1
            value_round *= 10
        return "$" + str(np.round(value_round, rounding)) + "$\n$\\times 10^{-" + str(pot) + "}$", pot
    
def plot_dict(begin_name, significant_val, dict_use, save_name, subtitle, use_var = []):
    plt.figure(figsize=(21*cm, 29.7/2.4*len(use_var)*cm), dpi = 300)
    
    plt.rcParams["svg.fonttype"] = "none"
    rc('font',**{'family':'Arial'})
    #plt.rcParams.update({"font.size": 7})
    SMALL_SIZE = 5
    MEDIUM_SIZE = 7
    BIGGER_SIZE = 7

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    ix_var = 0
    if len(use_var) == 0:
        use_var = list(dict_use.keys())
    for var in use_var:
        if var == "time":
            continue
        ix_ws = ix_var
        for ws in sorted(dict_use[var]):
            ix_ws += 1
            stepx = 1
            stepy = 1
            x = 0
            if len(dict_use[var][ws]) > 1:
                plt.subplot(2 * (len(use_var) - 1 * ("time" in use_var)), 4, ix_ws)
                for m1 in sorted(dict_use[var][ws]):
                    y = 0
                    for m2 in sorted(dict_use[var][ws]):
                        xarr = [x - stepx / 2, x + stepx / 2]
                        ymin = y - stepy / 2
                        ymax = y + stepy / 2
                        valu = 1 - dict_use[var][ws][m1][m2][1]
                        hexcode = "#" + str(hex(int(np.round(valu * 255, 0)))).replace("0x", "") * 3
                        mid_x = x - stepx / 2 + stepx / 16
                        if valu > 0.5:
                            hexcode_opposite = "#000000"
                        else:
                            hexcode_opposite = "#ffffff"
                        if len(dict_use[var][ws]) < 6:
                            label_hex = stringify(dict_use[var][ws][m1][m2][1], 3, False)[0]
                        else:
                            if dict_use[var][ws][m1][m2][1] > significant_val:
                                label_hex = "$\geq$"
                            else:
                                label_hex = "$<$"
                        if "\n" not in label_hex:
                            mid_y = (y - stepy / 2 + y + stepy / 2) / 2
                        else:
                            mid_y = (y - stepy / 2 + y + stepy / 2) / 2 - stepy / 4
                        plt.fill_between(xarr, ymin, ymax, color = hexcode)
                        if m1 != m2:
                            plt.text(mid_x, mid_y, label_hex, color = hexcode_opposite)
                        y += stepy
                    x += stepx
                range_vals = np.arange(- stepx / 2, x, stepx)
                for x2 in range_vals:
                    plt.axvline(x2, color = "k")
                    plt.axhline(x2, color = "k")
                plt.xlim(- stepx / 2, x - stepx / 2)
                plt.ylim(- stepy / 2, y - stepy / 2)
                ticks_use = [ix * stepx for ix in range(len(dict_use[var][ws]))]
                labs_use = [m.replace("_", " ") for m in sorted(dict_use[var][ws])]
                varnew = var.replace("_", " ").replace("longitude no abs", "$x$ offset").replace("direction", "heading")
                varnew = varnew.replace("latitude no abs", "$y$ offset").replace("no abs", "$x$ and $y$ offset")
                varnew = varnew.replace("speed actual dir", "speed, heading, and time")
                if ix_ws % 8 == 1 and len(use_var) > 1:
                    plt.ylabel(varnew.capitalize() + "\n" + subtitle)
                else:
                    if ix_ws % 8 == 2 and len(use_var) == 1:
                        plt.title(varnew.capitalize() + "\n" + subtitle)
                plt.xlabel("Forecasting time $" + str(ws) + "$ $s$")
                #if ix_ws % 8 == 7 and ix_var == 8 * (len(use_var) - 1 - 1 * ("time" in use_var)):
                plt.xticks(ticks_use, labs_use)
                plt.gca().xaxis.tick_top()
                if len(ticks_use) > 4:
                    plt.xticks(rotation = 90)
                plt.yticks([])
        ix_var += 8
    new_dir_name = "stats_dir/"    
    if begin_name == "dicti_wilcoxon":
        new_dir_name += "wilcoxon/"
    if begin_name == "dicti_mann_whitney":
        new_dir_name += "mann_whitney/"
    if not os.path.isdir(new_dir_name):
        os.makedirs(new_dir_name)
    plt.savefig(new_dir_name + save_name + ".svg", bbox_inches = "tight")
    plt.savefig(new_dir_name + save_name + ".png", bbox_inches = "tight")
    plt.savefig(new_dir_name + save_name + ".pdf", bbox_inches = "tight")
    plt.close()

name_list_total = ["data_frame_val_merged", "data_frame_traj_val_merged"]

round_val = {"R2": (100, 2, 3), "MAE": (1, 2, 1), "MSE": (1, 2, 1), "RMSE": (1, 2, 1), "euclid": (1, 2, 1), "haversine": (1, 2, 1)}
round_val = {"RMSE": (1, 2, 1)}
my_text_total = ""
my_appendix_total = ""
model_best_for = dict()
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
    
    model_best_for[name] = dict()
    for metric in round_val:
        if metric not in df_dictio:
            continue
        model_best_for[name][metric] = dict()
        for var in min_max_for_metric_for_ws:
            if "time" in var:
                continue
            model_best_for[name][metric][var] = dict()
            for model in dictio[var]:
                model_best_for[name][metric][var][model] = []
            for ws in min_max_for_metric_for_ws[var]:
                metric_min, model_min, metric_max, model_max = min_max_for_metric_for_ws[var][ws][metric]
                model_best = model_min
                if "R2" in metric:
                    model_best = model_max
                model_best_for[name][metric][var][model_best].append(ws)

#for start_name in ["dicti_mann_whitney", "dicti_wilcoxon"]:
for start_name in ["dicti_wilcoxon"]:

    #for metric in ["R2", "MAE", "RMSE", "MSE"]:
    for metric in ["RMSE"]:
        df_dictio = pd.read_csv(start_name + "_" + metric + ".csv", index_col = False)

        var_list = set(df_dictio["variable"])
        model_list1 = set(list(df_dictio["model1"]))
        model_list2 = set(list(df_dictio["model2"]))
        model_list = set()
        for m in model_list1:
            model_list.add(m)
        for m in model_list2:
            model_list.add(m)
        ws_list = set(df_dictio["ws"])

        dicti_mann_whitney = dict()

        for var in var_list:
            dicti_mann_whitney[var] = dict()
            for ws in ws_list:
                dicti_mann_whitney[var][ws] = dict()
                for model1 in model_list:
                    dicti_mann_whitney[var][ws][model1] = dict()
                    for model2 in model_list:
                        dicti_mann_whitney[var][ws][model1][model2] = (1.0, 1.0)

        found_sth = dict()
        found_sth_total = set()
        for ix in range(len(df_dictio["variable"])):
            var = df_dictio["variable"][ix]
            ws = df_dictio["ws"][ix]
            if var not in found_sth:
                found_sth[var] = dict()
            if ws not in found_sth[var]:
                found_sth[var][ws] = set()
            model1 = df_dictio["model1"][ix]
            model2 = df_dictio["model2"][ix]
            u = df_dictio["u"][ix] 
            p = df_dictio["p"][ix]
            len_use = len(dicti_mann_whitney[var][ws][model1])
            num_use = (len_use * (len_use + 1)) / 2
            for name_sth in model_best_for:
                if metric not in model_best_for[name_sth]:
                    continue
                if var not in model_best_for[name_sth][metric]:
                    continue
                if ws in model_best_for[name_sth][metric][var][model1]:
                    found_sth[var][ws].add(model1)
                    found_sth_total.add(model1)
                    if p >= 0.05 / num_use:
                        found_sth[var][ws].add(model2)
                        found_sth_total.add(model2)
                if ws in model_best_for[name_sth][metric][var][model2]:
                    found_sth[var][ws].add(model2)
                    found_sth_total.add(model2)
                    if p >= 0.05 / num_use:
                        found_sth[var][ws].add(model1)
                        found_sth_total.add(model1)
            dicti_mann_whitney[var][ws][model1][model2] = (u, p)
            dicti_mann_whitney[var][ws][model2][model1] = (u, p)

        dicti_mann_whitney_filter = dict()
        for var in dicti_mann_whitney:
            dicti_mann_whitney_filter[var] = dict()
            for ws in dicti_mann_whitney[var]:
                dicti_mann_whitney_filter[var][ws] = dict()
                print(metric, var, ws, len(found_sth[var][ws]))
                for m1 in found_sth[var][ws]:
                    dicti_mann_whitney_filter[var][ws][m1] = dict()
                    for m2 in found_sth[var][ws]:
                        dicti_mann_whitney_filter[var][ws][m1][m2] = dicti_mann_whitney[var][ws][m1][m2]
        
        dicti_mann_whitney_filter_total = dict()
        for var in dicti_mann_whitney:
            dicti_mann_whitney_filter_total[var] = dict()
            for ws in dicti_mann_whitney[var]:
                dicti_mann_whitney_filter_total[var][ws] = dict()
                for m1 in found_sth_total:
                    dicti_mann_whitney_filter_total[var][ws][m1] = dict()
                    for m2 in found_sth_total:
                        dicti_mann_whitney_filter_total[var][ws][m1][m2] = dicti_mann_whitney[var][ws][m1][m2]
        
        metricnew = metric.replace("R2", "$R^{2}$ (%)")
        metricnew = metricnew.replace("euclid", "Euclidean distance")
        metricnew = metricnew.replace("haversine", "Haversine distance")
        plot_dict(start_name, 0.05 / num_use, dicti_mann_whitney_filter, "var_long_" + metric, metricnew, ["longitude_no_abs"])
        plot_dict(start_name, 0.05 / num_use, dicti_mann_whitney_filter, "var_lat_" + metric, metricnew, ["latitude_no_abs"])
        plot_dict(start_name, 0.05 / num_use, dicti_mann_whitney_filter, "var_speed_" + metric, metricnew, ["speed"])
        plot_dict(start_name, 0.05 / num_use, dicti_mann_whitney_filter, "var_direction_" + metric, metricnew, ["direction"])
        #plot_dict(start_name, 0.05 / num_use, dicti_mann_whitney_filter_total, "var_long_lat_" + metric, metricnew, ["longitude_no_abs", "latitude_no_abs"])
        #plot_dict(start_name, 0.05 / num_use, dicti_mann_whitney_filter_total, "var_speed_direction_" + metric, metricnew, ["speed", "direction"])

    #for metric in ["R2", "MAE", "RMSE", "MSE", "euclid", "haversine"]:
    for metric in ["RMSE"]:
        df_dictio_traj = pd.read_csv(start_name + "_traj_" + metric + ".csv", index_col = False)

        var_list = set(df_dictio_traj["variable"])
        model_list1 = set(list(df_dictio_traj["model1"]))
        model_list2 = set(list(df_dictio_traj["model2"]))
        model_list = set()
        for m in model_list1:
            model_list.add(m)
        for m in model_list2:
            model_list.add(m)
        ws_list = set(df_dictio_traj["ws"])

        dicti_mann_whitney_traj = dict()

        for var in var_list:
            dicti_mann_whitney_traj[var] = dict()
            for ws in ws_list:
                dicti_mann_whitney_traj[var][ws] = dict()
                for model1 in model_list:
                    dicti_mann_whitney_traj[var][ws][model1] = dict()
                    for model2 in model_list:
                        dicti_mann_whitney_traj[var][ws][model1][model2] = (1.0, 1.0)

        found_sth = dict()
        found_sth_total = set()
        for ix in range(len(df_dictio_traj["variable"])):
            var = df_dictio_traj["variable"][ix]
            ws = df_dictio_traj["ws"][ix]
            if var not in found_sth:
                found_sth[var] = dict()
            if ws not in found_sth[var]:
                found_sth[var][ws] = set()
            model1 = df_dictio_traj["model1"][ix]
            model2 = df_dictio_traj["model2"][ix]
            u = df_dictio_traj["u"][ix] 
            p = df_dictio_traj["p"][ix] 
            len_use = len(dicti_mann_whitney_traj[var][ws][model1])
            num_use = (len_use * (len_use + 1)) / 2
            for name_sth in model_best_for:
                if metric not in model_best_for[name_sth]:
                    continue
                if var not in model_best_for[name_sth][metric]:
                    continue
                if ws in model_best_for[name_sth][metric][var][model1]:
                    found_sth[var][ws].add(model1)
                    found_sth_total.add(model1)
                    if p >= 0.05 / num_use:
                        found_sth[var][ws].add(model2)
                        found_sth_total.add(model2)
                if ws in model_best_for[name_sth][metric][var][model2]:
                    found_sth[var][ws].add(model2)
                    found_sth_total.add(model2)
                    if p >= 0.05 / num_use:
                        found_sth[var][ws].add(model1)
                        found_sth_total.add(model1)
            dicti_mann_whitney_traj[var][ws][model1][model2] = (u, p)
            dicti_mann_whitney_traj[var][ws][model2][model1] = (u, p)

        dicti_mann_whitney_traj_filter = dict()
        for var in dicti_mann_whitney_traj:
            dicti_mann_whitney_traj_filter[var] = dict()
            for ws in dicti_mann_whitney_traj[var]:
                dicti_mann_whitney_traj_filter[var][ws] = dict()
                print(metric, var, ws, len(found_sth[var][ws]))
                for m1 in found_sth[var][ws]:
                    dicti_mann_whitney_traj_filter[var][ws][m1] = dict()
                    for m2 in found_sth[var][ws]:
                        dicti_mann_whitney_traj_filter[var][ws][m1][m2] = dicti_mann_whitney_traj[var][ws][m1][m2]
        
        dicti_mann_whitney_traj_filter_total = dict()
        for var in dicti_mann_whitney_traj:
            dicti_mann_whitney_traj_filter_total[var] = dict()
            for ws in dicti_mann_whitney_traj[var]:
                dicti_mann_whitney_traj_filter_total[var][ws] = dict()
                for m1 in found_sth_total:
                    dicti_mann_whitney_traj_filter_total[var][ws][m1] = dict()
                    for m2 in found_sth_total:
                        dicti_mann_whitney_traj_filter_total[var][ws][m1][m2] = dicti_mann_whitney_traj[var][ws][m1][m2]
        
        metricnew = metric.replace("R2", "$R^{2}$ (%)")
        metricnew = metricnew.replace("euclid", "Euclidean distance")
        metricnew = metricnew.replace("haversine", "Haversine distance")
        plot_dict(start_name, 0.05 / num_use, dicti_mann_whitney_traj_filter, "traj_no_abs_" + metric, metricnew, ["no abs"])
        plot_dict(start_name, 0.05 / num_use, dicti_mann_whitney_traj_filter, "traj_speed_actual_dir_" + metric, metricnew, ["speed actual dir"])
        #plot_dict(start_name, 0.05 / num_use, dicti_mann_whitney_traj_filter_total, "traj_all_" + metric, metricnew, ["no abs", "speed actual dir"])