import pandas as pd
import numpy as np
import os

MAXINT = 10 ** 100
letters = ["A", "B", "C", "D", "E", "F"]
ix_letter = 0
num_tab_occured = dict()
all_text_for_tab = dict()
use_appendix = False
use_p_table = True
usep = True
useu = False
reverse_dir = False
print_var_res = False
print_ws_res = False
print_metric_res = True
print_pred_res = False
start_name = "dicti_wilcoxon"
translate_start_name = {"dicti_wilcoxon": "Wilcoxon signed-rank test", "dicti_mann_whitney": "Mann-Whitney $U$-test"}
def read_dict(dict_p_path, usable_cols):
    if not usep:
        return "", "", ""
    use_resizebox = len(usable_cols) > 4
    df_p_path = pd.read_csv(dict_p_path, index_col = False)
    if useu:
        df_u_path = pd.read_csv(dict_p_path.replace("/p_", "/u_"), index_col = False)
    label_name = "\\label{tab:" + dict_p_path.replace(start_name + "_", "").replace(start_name, "").replace("filter_", "").replace("/", "_").replace(" ", "_").replace(".csv", "") + "}"
    for word in ["ws", "base", "metric", "var"]:
        label_name = label_name.replace(word + "_", "")
        label_name = label_name.replace("_" + word, "")
    label_name = label_name.replace("_", ":")
    captext = "The $p$-values for the " + translate_start_name[start_name]  + " on "
    found_metric = False
    for metric in round_val:
        if metric in dict_p_path:
            metricnew = metric.replace("R2", "$R^{2}$ (\%)")
            metricnew = metricnew.replace("euclid", "Euclidean distance")
            metricnew = metricnew.replace("haversine", "haversine distance")
            captext += metricnew + " values across $k$-fold validation datasets for the $k$-fold testing datasets"
            found_metric = True
            break
    if not found_metric:
        captext += "actual and predicted values across $k$-fold validation datasets for the $k$-fold testing datasets"
    var_curr = ""
    for varn in var_list:
        if varn in dict_p_path:
            var_curr = varn.replace("_", " ").replace("longitude no abs", "$x$ offset").replace("direction", "heading")
            var_curr = var_curr.replace("latitude no abs", "$y$ offset").replace("no abs", "trajectories estimated using $x$ and $y$ offset")
            var_curr = var_curr.replace("speed actual dir", "trajectories estimated using speed, heading, and time")
    if "lon" in dict_p_path and "traj" in var_curr:
        var_curr = var_curr.replace("trajectories", "the longitude of trajectories")
    if "lat" in dict_p_path and "traj" in var_curr:
        var_curr = var_curr.replace("trajectories", "the latitude of trajectories")
    model_curr = ""
    for modeln in model_list:
        if modeln in dict_p_path:
            model_curr = modeln.replace("_", " ")
    ws_curr = ""
    for wsn in ws_list:
        if str(wsn) in dict_p_path:
            ws_curr = str(wsn)
    if df_p_path["Compare"][0] in model_list:
        captext += " using different RNN models, " + var_curr + ", and a forecasting time of $" + ws_curr + "$ $s$."
    if df_p_path["Compare"][0] in var_list:
        captext += " using different variables, the " + model_curr + " model, and a forecasting time of $" + ws_curr + "$ $s$."
    if df_p_path["Compare"][0] in ws_list:
        captext += " using different forecasting times, " + var_curr + ", and the " + model_curr + " model."
    start_latex = "\\begin{table}[!ht]\n\t\\centering" + "\n\t\\resizebox{\\linewidth}{!}{" * use_resizebox + "\n\t" + "\t" * use_resizebox + "\\begin{tabular}{|c" + "|c" * len(usable_cols) + "|}\n\t\t" + "\t" * use_resizebox + "\\hline\n"
    end_latex = "\t" + "\t" * use_resizebox + "\\end{tabular}" + "\n\t}" * use_resizebox + "\n\t\\caption{"+ captext + "}\n\t" + label_name + "\n\\end{table}"
    header_latex = "\t\t" + "\t" * use_resizebox + "Compare"
    if df_p_path["Compare"][0] in model_list:
        header_latex = "\t\t" + "\t" * use_resizebox + "Model"
    if df_p_path["Compare"][0] in var_list:
        header_latex = "\t\t" + "\t" * use_resizebox + "Variable"
    if df_p_path["Compare"][0] in ws_list:
        header_latex = "\t\t" + "\t" * use_resizebox + "Forecasting time"
    for colname in df_p_path["Compare"]:
        if colname not in usable_cols:
            continue
        if colname in model_list:
            colname = colname.replace("_", " ")
        if colname in var_list:
            colname = colname.replace("_", " ").replace("longitude no abs", "$x$ offset").replace("direction", "heading")
            colname = colname.replace("latitude no abs", "$y$ offset").replace("no abs", "$x$ and $y$ offset")
            colname = colname.replace("speed actual dir", "speed, heading, and time")
        if str(colname).isdigit():
            header_latex += " & $" + str(colname) + "$ $s$"
        else:
            header_latex += " & " + colname
    header_latex += " \\\\ \\hline\n"
    string_latex = ""
    for ix in range(len(df_p_path["Compare"])):
        first_value = df_p_path["Compare"][ix]
        if first_value not in usable_cols:
            continue
        str_pvals = "\t\t" + "\t" * use_resizebox
        if useu:
            str_pvals += "\\multirow{2}{*}{"
        if first_value in model_list:
            first_value = first_value.replace("_", " ")
        if first_value in var_list:
            first_value = first_value.replace("_", " ").replace("longitude no abs", "$x$ offset").replace("direction", "heading")
            first_value = first_value.replace("latitude no abs", "$y$ offset").replace("no abs", "$x$ and $y$ offset")
            first_value = first_value.replace("speed actual dir", "speed, heading, and time")
        if str(first_value).isdigit():
            str_pvals += "$" + str(first_value) + "$ $s$"
        else:
            str_pvals += str(first_value)
        if useu:
            str_pvals += "}"
        len_use = len(df_p_path["Compare"])
        num_use = (len_use * (len_use - 1)) / 2
        for colname in df_p_path["Compare"]:
            if colname not in usable_cols:
                continue
            if df_p_path["Compare"][ix] == colname:
                str_pvals += " & /"
                continue
            if (reverse_dir and df_p_path[str(colname)][ix] < 0.05 / num_use) or (not reverse_dir and df_p_path[str(colname)][ix] >= 0.05 / num_use):
                str_pvals += " & $\\mathbf{" + str(stringify(df_p_path[str(colname)][ix], 3 - 1 * reverse_dir, False)[0]) + "}$"
            else:
                str_pvals += " & $" + str(stringify(df_p_path[str(colname)][ix], 3 - 1 * reverse_dir, False)[0]) + "$"
        if not useu:
            str_pvals += " \\\\ \\hline\n"
        else:
            str_pvals += " \\\\\n"
        str_uvals = ""
        if useu:
            str_uvals = "\t\t" + "\t" * use_resizebox
            for colname in df_p_path["Compare"]:
                if colname not in usable_cols:
                    continue
                if df_p_path["Compare"][ix] == colname:
                    str_uvals += " & /"
                    continue
                if (reverse_dir and df_p_path[str(colname)][ix] < 0.05 / num_use) or (not reverse_dir and df_p_path[str(colname)][ix] >= 0.05 / num_use):
                    str_uvals += " & \\textbf{(}\\mathbf{$" + str(np.round(df_u_path[str(colname)][ix], 0))[:-2] + "}\\textbf{)}$)"
                else:
                    str_uvals += " & ($" + str(np.round(df_u_path[str(colname)][ix], 0))[:-2] + "$)"
            str_uvals += " \\\\ \\hline\n"
        string_latex += str_pvals + str_uvals
    return start_latex + header_latex + string_latex + end_latex, captext, label_name

def save_dict(some_dict, some_dict_file_dir, some_dict_file_path):
    if not os.path.isdir(some_dict_file_dir):
        os.makedirs(some_dict_file_dir)
    k1 = sorted(list(some_dict.keys()))
    save_long_lat = False
    if len(some_dict[k1[0]][k1[1]]) == 4:
        save_long_lat = True
    dict_csv_save_dict_u1 = {"Compare": []}
    dict_csv_save_dict_p1 = {"Compare": []}
    if save_long_lat:
        dict_csv_save_dict_u2 = {"Compare": []}
        dict_csv_save_dict_p2 = {"Compare": []}
    for v2 in k1:
        dict_csv_save_dict_u1["Compare"].append(v2)
        dict_csv_save_dict_p1["Compare"].append(v2)
        dict_csv_save_dict_u1[v2] = []
        dict_csv_save_dict_p1[v2] = []
        if save_long_lat:
            dict_csv_save_dict_u2["Compare"].append(v2)
            dict_csv_save_dict_p2["Compare"].append(v2)
            dict_csv_save_dict_u2[v2] = []
            dict_csv_save_dict_p2[v2] = []
    for v1 in k1:
        for v2 in k1:
            if v2 not in some_dict[v1]:
                dict_csv_save_dict_u1[v1].append(1.0)
                dict_csv_save_dict_p1[v1].append(1.0)
                if save_long_lat:
                    dict_csv_save_dict_u2[v1].append(1.0)
                    dict_csv_save_dict_p2[v1].append(1.0)
            else:
                dict_csv_save_dict_u1[v1].append(some_dict[v1][v2][0])
                dict_csv_save_dict_p1[v1].append(some_dict[v1][v2][1])
                if save_long_lat:
                    dict_csv_save_dict_u2[v1].append(some_dict[v1][v2][2])
                    dict_csv_save_dict_p2[v1].append(some_dict[v1][v2][3])

    if save_long_lat:
        dfu1 = pd.DataFrame(dict_csv_save_dict_u1)
        dfu1.to_csv(some_dict_file_dir + "u_long_" + some_dict_file_path, index = False)
        dfp1 = pd.DataFrame(dict_csv_save_dict_p1)
        dfp1.to_csv(some_dict_file_dir + "p_long_" + some_dict_file_path, index = False)
        dfu2 = pd.DataFrame(dict_csv_save_dict_u2)
        dfu2.to_csv(some_dict_file_dir + "u_lat_" + some_dict_file_path, index = False)
        dfp2 = pd.DataFrame(dict_csv_save_dict_p2)
        dfp2.to_csv(some_dict_file_dir + "p_lat_" + some_dict_file_path, index = False)
    else:
        dfu1 = pd.DataFrame(dict_csv_save_dict_u1)
        dfu1.to_csv(some_dict_file_dir + "u_" + some_dict_file_path, index = False)
        dfp1 = pd.DataFrame(dict_csv_save_dict_p1)
        dfp1.to_csv(some_dict_file_dir + "p_" + some_dict_file_path, index = False)
 
def print_sentence_metric(var, model, metric, ws):
    retval = ""
    list_of_entries = same_model_metric[metric][var][model][ws]
    if reverse_dir:
        list_of_entries = different_model_metric[metric][var][model][ws]
    if print_metric_res and len(list_of_entries) != 0:
        str_models = ""
        str_uvals = ""
        str_pvals = ""
        model_print = [model]
        for entry_one in list_of_entries:
            model2, u, p = entry_one
            model_print.append(model2)
            str_models += model2.replace("_", " ") + ", "
            str_uvals += "$" + str(np.round(u, 0))[:-2] + "$, "
            str_pvals += "$" + str(stringify(p, 3 - 1 * reverse_dir, False)[0]) + "$, "
        str_models = str_models[:-2]
        str_uvals = str_uvals[:-2]
        str_pvals = str_pvals[:-2]
        if "," in str_pvals:
            str_models = str_models.replace(str_models.split(",")[-1], " and" + str_models.split(",")[-1])
            str_uvals = str_uvals.replace(str_uvals.split(",")[-1], " and" + str_uvals.split(",")[-1])
            str_pvals = str_pvals.replace(str_pvals.split(",")[-1], " and" + str_pvals.split(",")[-1])
        varnew = var.replace("_", " ").replace("longitude no abs", "$x$ offset").replace("direction", "heading")
        varnew = varnew.replace("latitude no abs", "$y$ offset").replace("no abs", "$x$ and $y$ offset")
        varnew = varnew.replace("speed actual dir", "speed, heading, and time")
        metricnew = metric.replace("R2", "$R^{2}$ (\%)")
        metricnew = metricnew.replace("euclid", "Euclidean distance")
        metricnew = metricnew.replace("haversine", "haversine distance")
        sentence_use = "The " + model.replace("_", " ") + " model does not have a statistically significantly different " + metricnew + " than the " + str_models + " models for " + varnew + " using a forecasting time of $" + str(ws) + "$ $s$"
        ppart = ", with $p$-values equaling " + str_pvals
        upart = ", and statistics equaling " + str_uvals + " respectively"
        if usep:
            sentence_use += ppart
        if useu:
            sentence_use += upart
        if "," not in str_pvals:
            sentence_use = sentence_use.replace("models", "model").replace("$p$-values", "a $p$-value").replace("statistics", "a statistic")
        if reverse_dir:
            sentence_use = sentence_use.replace("does not have", "has")
        retval = sentence_use.replace("actual", "actual value") + ".\n\n"
        stat_test_results_file = "filter_" + start_name + "_base_metric/" + metric + "/" + var + "/p_" + str(ws) + "_" + start_name + "_base_metric.csv"
        stattab, capt, reflab = read_dict(stat_test_results_file, model_print)
        if use_appendix:
            retval += capt[:-1] + " are listed in Table~" + reflab.replace("label", "ref") + ".\n"
        if stat_test_results_file not in num_tab_occured:
            num_tab_occured[stat_test_results_file] = 0
            retval += "\\markertable{tab:" + reflab + "}\n\n"
        num_tab_occured[stat_test_results_file] += 1
        if stat_test_results_file not in all_text_for_tab:
            all_text_for_tab[stat_test_results_file] = set()
        for m in model_print:
            all_text_for_tab[stat_test_results_file].add(m)
    return retval
 
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
        return str(np.round(value_round, rounding)) + " \\times 10^{-" + str(pot) + "}", pot
    
name_list = ["data_frame_total_reverse", "data_frame_total", "data_frame_test", "data_frame_test_reverse", "data_frame_val"]
name_list_traj = [name.replace("frame_", "frame_traj_") for name in name_list]
name_list = ["data_frame_val_merged"]
name_list_traj = ["data_frame_traj_val_merged"]

name_list_total = []
name_list_total.extend(name_list)
name_list_total.extend(name_list_traj)

round_val = {"R2": (100, 2, 3), "MAE": (1, 2, 1), "MSE": (1, 2, 1), "RMSE": (1, 2, 1), "euclid": (1, 2, 1), "haversine": (1, 2, 1)}
round_val = {"RMSE": (1, 2, 1)}
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
    dicti_mann_whitney_var = dict()
    dicti_mann_whitney_ws = dict()
    dicti_mann_whitney_base_metric = dict()
    for metric in round_val:
        if metric not in df_dictio:
            continue
        if "euclid" in metric:
            continue
        model_best_for[metric] = dict() 
        dicti_mann_whitney_base_metric[metric] = dict()
        for var in min_max_for_metric_for_ws:
            if "time" in var:
                continue
            pd_file_base = start_name
            if " " in var:
                pd_file_base += "_traj"
            pd_file_var = pd_file_base + "_variables_" + metric + ".csv"
            pd_file_ws = pd_file_base + "_ws_" + metric + ".csv"
            pd_file_base_metric = pd_file_base + "_" + metric + ".csv"  
            df_pd_file_base_metric = pd.read_csv(pd_file_base_metric, index_col = False) 
            for ix_line in range(len(df_pd_file_base_metric)):
                var_use = df_pd_file_base_metric["variable"][ix_line]
                ws_use = df_pd_file_base_metric["ws"][ix_line]
                model1 = df_pd_file_base_metric["model1"][ix_line]
                model2 = df_pd_file_base_metric["model2"][ix_line]
                u = df_pd_file_base_metric["u"][ix_line]
                p = df_pd_file_base_metric["p"][ix_line]
                if var_use not in dicti_mann_whitney_base_metric[metric]:
                    dicti_mann_whitney_base_metric[metric][var_use] = dict()
                if ws_use not in dicti_mann_whitney_base_metric[metric][var_use]:
                    dicti_mann_whitney_base_metric[metric][var_use][ws_use] = dict()
                if model1 not in dicti_mann_whitney_base_metric[metric][var_use][ws_use]:
                    dicti_mann_whitney_base_metric[metric][var_use][ws_use][model1] = dict()
                if model2 not in dicti_mann_whitney_base_metric[metric][var_use][ws_use]:
                    dicti_mann_whitney_base_metric[metric][var_use][ws_use][model2] = dict()
                dicti_mann_whitney_base_metric[metric][var_use][ws_use][model1][model2] = (u, p)
                dicti_mann_whitney_base_metric[metric][var_use][ws_use][model2][model1] = (u, p)
            model_best_for[metric][var] = dict()
            for model in dictio[var]:
                model_best_for[metric][var][model] = []
            for ws in min_max_for_metric_for_ws[var]:
                metric_min, model_min, metric_max, model_max = min_max_for_metric_for_ws[var][ws][metric]
                model_best = model_min
                if "R2" in metric:
                    model_best = model_max
                model_best_for[metric][var][model_best].append(ws) 
  
    different_model_metric = dict()
    same_model_metric = dict()
    for metric in round_val:
        if metric not in df_dictio:
            continue
        if "euclid" in metric:
            continue 
        different_model_metric[metric] = dict()
        same_model_metric[metric] = dict()
        for var in dictio:
            if "time" in var:
                continue 
            different_model_metric[metric][var] = dict()
            same_model_metric[metric][var] = dict()
            for model in sorted((dictio[var].keys())): 
                different_model_metric[metric][var][model] = dict()
                same_model_metric[metric][var][model] =dict()
                for ws in sorted((dictio[var][model].keys())): 
                    different_model_metric[metric][var][model][ws]  = []
                    same_model_metric[metric][var][model][ws]  = []
                    for model2 in dicti_mann_whitney_base_metric[metric][var][ws][model]:
                        u, p = dicti_mann_whitney_base_metric[metric][var][ws][model][model2]
                        len_use = len(dicti_mann_whitney_base_metric[metric][var][ws][model])
                        num_use = (len_use * (len_use + 1)) / 2
                        if p < 0.05 / num_use:
                            different_model_metric[metric][var][model][ws].append((model2, u, p))
                        else:
                            same_model_metric[metric][var][model][ws].append((model2, u, p))
                    
    for metric in sorted(list(round_val.keys())):
        if metric not in df_dictio:
            continue
        if "euclid" in metric:
            continue 
        for var in sorted(var_list):
            if "time" in var:
                continue 
            for ws in sorted(ws_list):
                save_dict(dicti_mann_whitney_base_metric[metric][var][ws], "filter_" + start_name + "_base_metric/" + metric + "/" + var + "/", str(ws) + "_" + start_name + "_base_metric.csv")
 
    my_text = ""
    num_tab_occured = dict()
    all_text_for_tab = dict()
    metric_occured = set()
    if "traj" in name:
        print_metric_res = True
        usep = True
        use_p_table = True
    else:
        print_metric_res = True
        usep = True
        use_p_table = True
    for var in min_max_for_metric_for_ws:
        if "time" in var:
            continue
        varnew = var.replace("_", " ").replace("longitude no abs", "$x$ offset").replace("direction", "heading")
        varnew = varnew.replace("latitude no abs", "$y$ offset").replace("no abs", "$x$ and $y$ offset")
        varnew  = varnew.replace("speed actual dir", "speed, heading, and time")
        varnewsubtitle = var.replace("_", " ").replace("longitude no abs", "$x$ Offset").replace("direction", "Heading")
        varnewsubtitle = varnewsubtitle.replace("latitude no abs", "$y$ Offset").replace("no abs", "Trajectories Estimated Using $x$ and $y$ Offset")
        varnewsubtitle = varnewsubtitle.replace("speed actual dir", "Trajectories Estimated Using Speed, Heading, and Time").replace("speed", "Speed")
        subtitle = "\\subsection{Results for " + varnewsubtitle + "}\n%\\label{subsec:" + var.replace(" ", "_") + "_results}\n%\\vspace{10pt}\n"
        my_text_var = subtitle + "\n"
        if use_p_table:
            var_fig = var.replace("_", " ").replace("longitude no abs", "$x$ offset").replace("direction", "heading")
            var_fig = var_fig.replace("latitude no abs", "$y$ offset").replace("no abs", "trajectories estimated using $x$ and $y$ offset")
            var_fig = var_fig.replace("speed actual dir", "trajectories estimated using speed, heading, and time")
            sentence_add = " Darker colors in grayscale represent a higher $p$-value in a range from $0$ to $1$. The values on the secondary diagonal are all equal to $1$ and black because models equal themselves."
            if "traj" in name:
                label_mod = "traj_" + var.replace(" ", "_")
            else:    
                label_mod = "var_" + var.replace("itude_no_abs", "")
            text_mod = "$p$-values for the " + translate_start_name[start_name]  + " on RMSE values across $k$-fold validation datasets for the " + var_fig + " in the $k$-fold testing datasets using different RNN models, and forecasting times."
            my_text_var += "\n\n\nFigure~\\ref{fig:" + label_mod + "_RMSE} represents the " + text_mod + sentence_add
            my_text_var += "\n\n\\begin{figure}[!ht]\n\t\\centering\n\t\\includegraphics[width = 0.99 \\linewidth]{" + label_mod + "_RMSE.pdf}"
            my_text_var += "\n\t\\caption{The " + text_mod + sentence_add + "}\n\t\\label{fig:" + label_mod + "_RMSE}\n\\end{figure}\n\n\n"
        for metric in round_val:
            if metric not in df_dictio:
                continue
            if "euclid" in metric:
                continue
            if metric not in metric_occured:
                metricfigure = metric.replace("R2", "$R^{2}$ (\%)")
                metricfigure = metricfigure.replace("euclid", "Euclidean distance in $\\degree$")
                metricfigure = metricfigure.replace("haversine", "haversine distance in $km$")
                text_mod = "average " + metricfigure + " across $k$-fold testing datasets using different validation datasets for all variables estimated in nested $k$-fold cross-validation by different RNN models, and forecasting times."
                if "traj" in name:
                    text_mod = text_mod.replace("variables", "trajectories").replace("different", "different trajectory estimation methods,")
                    label_mod = "wilcoxon_" + metric + "_traj_val"
                else:
                    label_mod = "wilcoxon_" + metric + "_val"
                my_text_var += "\n\n\nFigure~\\ref{fig:" + label_mod + "_merged} contains the " + text_mod
                my_text_var += "\n\n\\begin{figure}[!ht]\n\t\\centering\n\t\\includegraphics[width = 0.99 \\linewidth]{" + label_mod + "_merged.pdf}"
                my_text_var += "\n\t\\caption{The " + text_mod + "}\n\t\\label{fig:" + label_mod + "_merged}\n\\end{figure}\n\n\n"
                metric_occured.add(metric)
            texfilepath = "tex_new_dir/" + name + "/wilcoxon_" + var.replace(" ", "_") + "_" + metric + ".tex"
            texfile = open(texfilepath, "r")
            texlines = texfile.readlines()
            linestex = ""
            texcaption = ""
            texlab = ""
            for l in texlines:
                linestex += l
                if "caption" in l:
                    texcaption = l.strip().replace("\\caption{", "")
                if "label" in l:
                    texlab = l.strip().replace("label", "ref")
            my_text_var += texcaption.replace(".}", " is listed in Table~" + texlab + ".\n\n") + linestex + "\n\n"
            model_best_for[metric][var][model] = sorted(model_best_for[metric][var][model])
            for model in sorted(list(model_best_for[metric][var].keys())):
                if len(model_best_for[metric][var][model]) > 0:
                    mul_val = 0
                    direction_use = "lowest"
                    if metric == "R2":
                        mul_val = 2
                        direction_use = "highest"
                    if metric == "MAE" and "no_abs" in var:
                        mul_val = 5
                    if metric == "MAE" and ("no abs" in var or "actual" in var):
                        mul_val = 3
                    if metric == "RMSE" and "no_abs" in var:
                        mul_val = 4
                    if metric == "RMSE" and "no abs" in var:
                        mul_val = 3
                    if metric == "RMSE" and "actual" in var:
                        mul_val = 2
                    rnd_val = 2
                    if metric == "MAE" and "latitude" in var:
                        rnd_val = 3
                    if metric == "RMSE" and ("no_abs" in var or "actual" in var):
                        rnd_val = 3
                    if metric == "haversine":
                        rnd_val = 3
                    mul_str = ""
                    if mul_val > 0 and "R2" != metric:
                        mul_str = " \\times 10^{-" + str(mul_val) + "}"
                    avg_arr = [np.round(dictio_avg[var][model][ws][metric] * (10 ** mul_val), rnd_val) for ws in model_best_for[metric][var][model]]
                    std_arr = [np.round(dictio_stdev[var][model][ws][metric] * (10 ** mul_val), rnd_val) for ws in model_best_for[metric][var][model]]
                    varnewnew = varnew
                    if "actual" in var or "no abs" in var:
                        varnewnew = "trajectories estimated using " + varnewnew
                    str_ws = ""
                    for ws in model_best_for[metric][var][model]:
                        str_ws += "$" + str(ws) + "$, "
                    str_ws = str_ws[:-2]    
                    if "," in str_ws:
                        str_ws = str_ws.replace(", $" + str(model_best_for[metric][var][model][-1]) + "$", ", and $" + str(model_best_for[metric][var][model][-1]) + "$")
                    str_avg_std = ""
                    last_part = ""
                    for ix in range(len(model_best_for[metric][var][model])):
                        unit_str = "\\%"
                        if metric == "haversine":
                            unit_str = " $km$"
                        if "euclid" in metric or (metric in ["MAE", "RMSE"] and ("heading" in varnew or "offset" in varnew)):
                            unit_str = " $\\degree$"
                        if metric in ["MAE", "RMSE"] and "speed" == varnew:
                            unit_str = " $km/h$"
                        last_part = " $" + str(avg_arr[ix]) + mul_str + "$" + unit_str + " ($" + str(std_arr[ix]) + mul_str + "$" + unit_str + ")"
                        str_avg_std += last_part + ","
                    str_avg_std = str_avg_std[1:-1]
                    if "," in str_avg_std:
                        str_avg_std = str_avg_std.replace(last_part, " and" + last_part)
                    metricnew = metric.replace("R2", "$R^{2}$ (\%)")
                    metricnew = metricnew.replace("euclid", "Euclidean distance")
                    metricnew = metricnew.replace("haversine", "haversine distance")
                    sentence_use = "The " + model.replace("_", " ") + " model achieved the " + direction_use + " " + metricnew + " for " + varnewnew + ", and a forecasting time of " + str_ws + " $s$ with average values and standard deviation (in brackets) that equal " + str_avg_std
                    if "," not in str_ws:
                        sentence_use = sentence_use.replace("average values", "an average value")
                        sentence_use = sentence_use.replace("equal", "equals")
                    if "," in str_ws:
                        sentence_use = sentence_use + " respectively"
                    my_text_var += sentence_use + ".\n\n"
                    for ws in model_best_for[metric][var][model]: 
                        my_text_var += print_sentence_metric(var, model, metric, ws) 
                        my_text_var += "\n"
        while "\n\n\n" in my_text_var:
            my_text_var = my_text_var.replace("\n\n\n", "\n\n")
        filenewtext = "latex_new_" + name + "_" + var.replace(" ", "_") + ".tex"
        filenewtextopened = open(filenewtext, "w")
        filenewtextopened.write(my_text_var)
        filenewtextopened.close()
        my_text += my_text_var
    marker_table_list = []
    my_appendix = "\\appendix\n\\section{Appendix " + letters[ix_letter] + "}\n\\label{app" + letters[ix_letter] + "}\n\n"
    ix_letter += 1
    for r in all_text_for_tab:
        stattab, capt, reflab = read_dict(r, all_text_for_tab[r]) 
        my_text = my_text.replace("\\markertable{tab:" + reflab + "}", "")
        my_appendix += capt[:-1] + " are listed in Table~" + reflab.replace("label", "ref") + ".\n\n"
        my_appendix += stattab + "\n\n"
    while "\n\n\n" in my_text:
        my_text = my_text.replace("\n\n\n", "\n\n")
    while "\n\n\n" in my_appendix:
        my_appendix = my_appendix.replace("\n\n\n", "\n\n")
    filenewtext = "latex_new_" + name + ".tex"
    filenewtextopened = open(filenewtext, "w")
    filenewtextopened.write(my_text)
    filenewtextopened.close()
    my_text_total += my_text
    filenewtextappendix = "latex_new_" + name + "_appendix.tex"
    filenewtextopenedappendix = open(filenewtextappendix, "w")
    filenewtextopenedappendix.write(my_appendix)
    filenewtextopenedappendix.close()
    my_appendix_total += my_appendix
    only_best = True
    for metric in sorted(list(round_val.keys())):
        if metric not in df_dictio:
            continue
        if "euclid" in metric:
            continue
        for var in sorted(var_list):
            if "time" in var:
                continue
            if only_best:
                use_model = list(model_best_for[metric][var].keys())
            else:
                use_model = model_list
            for model in sorted(use_model):
                if only_best:
                    use_ws = model_best_for[metric][var][model]
                else:
                    use_ws = ws_list
                for ws in sorted(use_ws): 
                    print_sentence_metric(var, model, metric, ws)
                    
my_text_total_file = open("my_text_total.tex", "w")
my_text_total_file.write(my_text_total)
my_text_total_file.close()
my_text_total_fileappendix = open("my_appendix_total.tex", "w")
my_text_total_fileappendix.write(my_appendix_total)
my_text_total_fileappendix.close()