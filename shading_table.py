import pandas as pd
import numpy as np
import os

MAXINT = 10 ** 100
cm = 1/2.54  # centimeters in inches

def stringify(value_round, rounding, skip_mul):
    if value_round == 0:
        return "$0$", 0
    if abs(value_round) >= 1 or skip_mul:
        return "$" + str(np.round(value_round, rounding)) + "$", 0
    else:
        pot = 0
        while abs(value_round) < 1:
            pot += 1
            value_round *= 10
        return "$" + str(np.round(value_round, rounding)) + "$\n$\\times 10^{-" + str(pot) + "}$", pot

def place_in_board_on_pos(dimensions, places):
    maxx = max([places[ws][0] + dimensions[ws][0] for ws in places])
    maxy = max([places[ws][1] + dimensions[ws][1] for ws in places])
    board_current = [[0 for y in range(maxy)] for x in range(maxx)]
    for ws in places:
        x_place, y_place = places[ws]
        for x2 in range(x_place, x_place + dimensions[ws][0]):
            for y2 in range(y_place, y_place + dimensions[ws][1]):
                board_current[x2][y2] = ws
    return board_current
 
def plot_dict(begin_name, significant_val, dict_use, save_name, subtitle, use_var = []):
    if len(use_var) == 0:
        use_var = list(dict_use.keys())
    string_for_var = dict()
    dict_sizes = dict()
    for var in use_var:
        if var == "time":
            continue
        string_for_var[var] = dict()
        dict_sizes[var] = dict()
        for ws in sorted(dict_use[var]):
            if len(dict_use[var][ws]) > 1:
                string_for_var[var][ws] = [["\\multicolumn{" + str(len(dict_use[var][ws])) + "}{c}{$" + str(ws) + "$ $s$}"]]
                line_one = []
                line_two = []
                for m1 in sorted(dict_use[var][ws]):
                    line_total = m1.replace("_", " ").replace("GRU ", "GRU\n").replace("RNN ", "RNN\n").replace("LSTM ", "LSTM\n")
                    if "\n" in line_total:
                        parts = line_total.split("\n")
                    else:
                        parts = ["\\multirow{2}{*}{" + line_total + "}", "XXXXXXXXXX"]
                    line_one.append(parts[0])
                    line_two.append(parts[1])
                string_for_var[var][ws].append(line_one)
                string_for_var[var][ws].append(line_two)
                for m1 in sorted(dict_use[var][ws]):
                    line_one = []
                    line_two = []
                    for m2 in sorted(dict_use[var][ws]):
                        label_hex = stringify(dict_use[var][ws][m1][m2][1], 3, False)[0]
                        if dict_use[var][ws][m1][m2][1] > significant_val and m1 != m2:
                            if "\n" in label_hex:
                                parts = label_hex.split("\n")
                                parts = ["$\\mathbf{" + parts[0][1:-1] + "}$", "$\\mathbf{" + parts[1][1:-1] + "}$"]
                            else:
                                parts = ["\\multirow{2}{*}{$\\mathbf{" + label_hex[1:-1] + "}$}", "XXXXXXXXXX"]
                        else:
                            if "\n" in label_hex:
                                parts = label_hex.split("\n")
                            else:
                                parts = ["\\multirow{2}{*}{" + label_hex + "}", "XXXXXXXXXX"]
                        line_one.append(parts[0])
                        line_two.append(parts[1])
                    string_for_var[var][ws].append(line_one)
                    string_for_var[var][ws].append(line_two)
                dict_sizes[var][ws] = (len(string_for_var[var][ws]), len(dict_use[var][ws]))
        
        if "speed" in var and "dir" not in var:
            p = {2: (0, 0), 3: (13, 0), 4: (0, 5), 5: (9, 5), 10: (0, 8), 20: (9, 8), 30: (18, 8)}
        if "dir" in var and "speed" not in var:
            p = {2: (0, 0), 3: (7, 0), 4: (0, 2), 5: (13, 2), 10: (0, 7), 20: (9, 7), 30: (18, 7)}
        if "lat" in var:
            p = {2: (0, 0), 3: (23, 0), 4: (0, 10), 5: (23, 14), 10: (36, 14), 20: (45, 14), 30: (36, 17)}
        if "long" in var:
            p = {2: (0, 0), 3: (15, 0), 4: (0, 6), 5: (15, 11), 10: (0, 12), 20: (7, 12), 30: (0, 14)}
        if "abs" in var and "lon" not in var and "lat" not in var:
            p = {2: (0, 0), 3: (0, 4), 4: (9, 4), 5: (0, 7)}
        if "dir" in var and "speed" in var:
            p = {2: (0, 0), 3: (11, 0), 4: (0, 4), 5: (11, 4), 10: (0, 8), 20: (0, 10), 30: (11, 10)}
        b = place_in_board_on_pos(dict_sizes[var], p)

        #for b1 in b:
            #print(b1)
        
        is_line = []
        for x2 in range(len(b)):
            is_line.append([])
            for y2 in range(len(b[x2])):
                is_line[-1].append(0)
        for ws in p:
            for x2 in range(p[ws][0], p[ws][0] + dict_sizes[var][ws][0]):
                for y2 in range(p[ws][1], p[ws][1] + dict_sizes[var][ws][1]):
                    if x2 - p[ws][0] >= 0 and y2 - p[ws][1] >= 0 and x2 - p[ws][0] < len(string_for_var[var][ws]) and y2 - p[ws][1] < len(string_for_var[var][ws][x2 - p[ws][0]]):
                        b[x2][y2] = string_for_var[var][ws][x2 - p[ws][0]][y2 - p[ws][1]]
                        if "1.0" not in b[x2][y2] and not ("$" in b[x2][y2] and not "times" in b[x2][y2] and not "s" in b[x2][y2]) and "UniTS" not in b[x2][y2] and "Conv" not in b[x2][y2] and "Bi" not in b[x2][y2] and "RNN" not in b[x2][y2] and "GRU" not in b[x2][y2] and "LSTM" not in b[x2][y2]:
                            is_line[x2][y2] = 1
                        if x2 + 1 < len(b) and "$s$" in str(b[x2 + 1][y2]):
                            is_line[x2][y2] = 1
                        if "$s$" in b[x2][y2]:
                            is_line[x2][y2] = 1
                    else:
                        is_line[x2 - 1][y2] = 1
                        is_line[x2][y2] = 1

        clines = ""
        for num_ix in range(len(b[0])):
            if b[0][num_ix] != 0:
                clines += " \\cline{" + str(num_ix + 1) + "-" + str(num_ix + 1) + "}"

        for x2 in range(len(b)):
            for y2 in range(len(b[x2])):
                if b[x2][y2] == 0:
                    b[x2][y2] = " "

        use_resizebox = True
        usable_cols = max([len(b[ixw]) for ixw in range(len(b))]) - 1
        var_fig = var.replace("_", " ").replace("longitude no abs", "$x$ offset").replace("direction", "heading")
        var_fig = var_fig.replace("latitude no abs", "$y$ offset").replace("no abs", "trajectories estimated using $x$ and $y$ offset")
        var_fig = var_fig.replace("speed actual dir", "trajectories estimated using speed, heading, and time")
        sentence_add = " The values on the primary diagonal are omitted because models equal themselves."

        translate_start_name = {"dicti_wilcoxon": "Wilcoxon signed-rank test", "dicti_mann_whitney": "Mann-Whitney $U$-test"}
        if "traj" in subtitle:
            label_mod = "traj_" + var.replace(" ", "_")
        else:    
            label_mod = "var_" + var.replace("itude_no_abs", "")
        text_mod = "$p$-values for the " + translate_start_name[start_name]  + " on RMSE values across $k$-fold validation datasets for the " + var_fig + " in the $k$-fold testing datasets using different RNN models, and forecasting times."
        my_text_var = "\nTable~\\ref{tab:" + label_mod + "_RMSE} represents the " + text_mod + sentence_add
        start_latex = "\n\n\\begin{table}[!ht]\n\t\\centering" + "\n\t\\resizebox{\\linewidth}{!}{" * use_resizebox + "\n\t" + "\t" * use_resizebox + "\\begin{tabular}{|c" + "|c" * usable_cols + "|}\n\t\t" + "\t" * use_resizebox + clines[1:] + "\n"
        for num in range(usable_cols + 2):
            start_latex = start_latex.replace("-" + str(num) + "} \\cline{" + str(num + 1),  "")
        end_latex = "\t" + "\t" * use_resizebox + "\\end{tabular}" + "\n\t}" * use_resizebox + "\n\t\\caption{The " + text_mod + sentence_add + "}\n\t\\label{tab:" + label_mod + "_RMSE}\n\\end{table}"
        strpr = my_text_var + start_latex
        for rix in range(len(b)):
            rn = [i for i in b[rix] if i not in p]
            clines = ""
            for num_ix in range(len(is_line[rix])):
                if is_line[rix][num_ix]:
                    clines += " \\cline{" + str(num_ix + 1) + "-" + str(num_ix + 1) + "}"
            strprpart = str(rn).replace("', '", " & ").replace("']", "\\\\ " + clines).replace("['", "\t\t" + use_resizebox * "\t")
            while "  " in strprpart:
                strprpart = strprpart.replace("  ", " ")
            strprpart = strprpart.replace("\\\\m", "\\m").replace("\\\\t", "\\t").replace("}\\", "} \\").replace("&\\", "& \\")
            for num_cols in range(len(is_line[rix]) + 4, 2, -1):
                pattern = "& " * num_cols
                strprpart = strprpart.replace(pattern[:-1], "& \\multicolumn{" + str(num_cols - 1) + "}{c}{} &")
                strprpart = strprpart.replace("\t" + pattern[1:-1], "\t \\multicolumn{" + str(num_cols - 1) + "}{c}{} &")
                strprpart = strprpart.replace(pattern[:-2] + "\\\\ ", "& \\multicolumn{" + str(num_cols - 1) + "}{c}{} \\\\ ")
                strprpart = strprpart.replace("\t" + pattern[1:-2] + "\\\\ ", "\t \\multicolumn{" + str(num_cols - 1) + "}{c}{} \\\\ ")
            for num in range(usable_cols + 2):
                strprpart = strprpart.replace("-" + str(num) + "} \\cline{" + str(num + 1),  "")
            strprpart = strprpart.replace("$1.0$", "")
            while "  " in strprpart:
                strprpart = strprpart.replace("  ", " ")
            strprpart = strprpart.replace("XXXXXXXXXX", "").replace("\\\\m", "\\m").replace("\\\\t", "\\t").replace("}\\", "} \\").replace("&\\", "& \\")
            while "  " in strprpart:
                strprpart = strprpart.replace("  ", " ")
            strpr += strprpart + "\n"
    new_dir_name = "stats_dir_table/"    
    if begin_name == "dicti_wilcoxon":
        new_dir_name += "wilcoxon/"
    if begin_name == "dicti_mann_whitney":
        new_dir_name += "mann_whitney/"
    if not os.path.isdir(new_dir_name):
        os.makedirs(new_dir_name)
    filenewtext =new_dir_name + save_name + ".tex"
    filenewtextopened = open(filenewtext, "w")
    filenewtextopened.write(strpr + end_latex)
    filenewtextopened.close()

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