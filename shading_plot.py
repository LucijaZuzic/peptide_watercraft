import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

MAXINT = 10 ** 100
cm = 1/2.54  # centimeters in inches

def plot_dict(begin_name, dict_use, save_name, use_var = []):
    plt.figure(figsize=(21*cm, 29.7/2.4*len(use_var)*cm), dpi = 300)
    
    plt.rcParams["svg.fonttype"] = "none"
    rc('font',**{'family':'Arial'})
    #plt.rcParams.update({"font.size": 7})
    SMALL_SIZE = 7
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
            plt.subplot(2 * (len(use_var) - 1 * ("time" in use_var)), 4, ix_ws)
            stepx = 1
            stepy = 1
            x = 0
            for m1 in sorted(dict_use[var][ws]):
                y = 0
                for m2 in sorted(dict_use[var][ws]):
                    xarr = [x - stepx / 2, x + stepx / 2]
                    ymin = y - stepy / 2
                    ymax = y + stepy / 2
                    valu = 1 - dict_use[var][ws][m1][m2][1]
                    hexcode = "#" + str(hex(int(np.round(valu * 255, 0)))).replace("0x", "") * 3
                    plt.fill_between(xarr, ymin, ymax, color = hexcode)
                    y += stepy
                x += stepx
            range_vals = np.arange(- stepx / 2, x, stepx)
            for x2 in range_vals:
                plt.axvline(x2, color = "k")
                plt.axhline(x2, color = "k")
            plt.xticks(rotation = 90)
            plt.xlim(- stepx / 2, x - stepx / 2)
            plt.ylim(- stepy / 2, y - stepy / 2)
            ticks_use = [ix * stepx for ix in range(len(dict_use[var][ws]))]
            labs_use = [m.replace("_", " ") for m in sorted(dict_use[var][ws])]
            varnew = var.replace("_", " ").replace("direction", "heading")
            varnew = varnew.replace("latitude speed", "latitude - speed").replace("longitude speed", "longitude - speed")
            varnew = varnew.replace("speed actual dir", "speed, heading, and time")
            if "traj" in save_name:
                varnew = varnew.replace("no abs", "- $x$ and $y$ offset")
            else:
                varnew = varnew.replace("no abs", "")
                varnew = varnew.replace("longitude", "$x$ offset")
                varnew = varnew.replace("latitude", "$y$ offset")
            if ix_ws % 8 == 1 and len(use_var) > 1:
                plt.ylabel(varnew.capitalize())
            else:
                if ix_ws % 8 == 2 and len(use_var) == 1:
                    plt.title(varnew.capitalize())
            plt.xlabel("Forecasting time $" + str(ws) + "$ $s$")
            if ix_ws % 8 == 7 and ix_var == 8 * (len(use_var) - 1 - 1 * ("time" in use_var)):
                plt.yticks(ticks_use, labs_use)
                plt.gca().yaxis.tick_right()
            else:
                plt.yticks([])
            plt.xticks([])
        ix_var += 8
    new_dir_name = "stats_new_dir/"    
    if begin_name == "dicti_wilcoxon":
        new_dir_name += "wilcoxon/"
    if not os.path.isdir(new_dir_name):
        os.makedirs(new_dir_name)
    plt.savefig(new_dir_name + save_name + ".svg", bbox_inches = "tight")
    plt.savefig(new_dir_name + save_name + ".png", bbox_inches = "tight")
    plt.savefig(new_dir_name + save_name + ".pdf", bbox_inches = "tight")
    plt.close()

for start_name in ["dicti_wilcoxon"]:

    df_dictio = pd.read_csv(start_name + "_merged.csv", index_col = False)
    df_dictio_traj = pd.read_csv(start_name + "_traj_merged.csv", index_col = False)

    var_list = set(df_dictio["variable"])
    traj_list = set(df_dictio_traj["variable"])
    model_list1 = set(list(df_dictio_traj["model1"]))
    model_list2 = set(list(df_dictio_traj["model2"]))
    model_list = set()
    for m in model_list1:
        model_list.add(m)
    for m in model_list2:
        model_list.add(m)
    ws_list = set(df_dictio_traj["ws"])

    dicti_mann_whitney = dict()

    for var in var_list:
        dicti_mann_whitney[var] = dict()
        for ws in ws_list:
            dicti_mann_whitney[var][ws] = dict()
            for model1 in model_list:
                dicti_mann_whitney[var][ws][model1] = dict()
                for model2 in model_list:
                    dicti_mann_whitney[var][ws][model1][model2] = (1.0, 1.0)

    for ix in range(len(df_dictio["variable"])):
        var = df_dictio["variable"][ix]
        ws = df_dictio["ws"][ix]
        model1 = df_dictio["model1"][ix]
        model2 = df_dictio["model2"][ix]
        u = df_dictio["u"][ix] 
        p = df_dictio["p"][ix]
        dicti_mann_whitney[var][ws][model1][model2] = (u, p)
        dicti_mann_whitney[var][ws][model2][model1] = (u, p)

    plot_dict(start_name, dicti_mann_whitney, "var_long", ["longitude_no_abs"])
    plot_dict(start_name, dicti_mann_whitney, "var_lat", ["latitude_no_abs"])
    plot_dict(start_name, dicti_mann_whitney, "var_speed", ["speed"])
    plot_dict(start_name, dicti_mann_whitney, "var_direction", ["direction"])
    plot_dict(start_name, dicti_mann_whitney, "var_long_lat", ["longitude_no_abs", "latitude_no_abs"])
    plot_dict(start_name, dicti_mann_whitney, "var_speed_direction", ["speed", "direction"])

    dicti_mann_whitney_traj = dict()

    for traj in traj_list:
        dicti_mann_whitney_traj["longitude " + traj] = dict()
        dicti_mann_whitney_traj["latitude " + traj] = dict()
        for ws in ws_list:
            dicti_mann_whitney_traj["longitude " + traj][ws] = dict()
            dicti_mann_whitney_traj["latitude " + traj][ws] = dict()
            for model1 in model_list:
                dicti_mann_whitney_traj["longitude " + traj][ws][model1] = dict()
                dicti_mann_whitney_traj["latitude " + traj][ws][model1] = dict()
                for model2 in model_list:
                    dicti_mann_whitney_traj["longitude " + traj][ws][model1][model2] = (1.0, 1.0)
                    dicti_mann_whitney_traj["latitude " + traj][ws][model1][model2] = (1.0, 1.0)

    for ix in range(len(df_dictio_traj["variable"])):
        var = df_dictio_traj["variable"][ix]
        ws = df_dictio_traj["ws"][ix]
        model1 = df_dictio_traj["model1"][ix]
        model2 = df_dictio_traj["model2"][ix]
        ulong = df_dictio_traj["u long"][ix] 
        plong = df_dictio_traj["p long"][ix] 
        ulat = df_dictio_traj["u lat"][ix] 
        plat = df_dictio_traj["p lat"][ix] 
        dicti_mann_whitney_traj["longitude " + var][ws][model1][model2] = (ulong, plong)
        dicti_mann_whitney_traj["longitude " + var][ws][model2][model1] = (ulong, plong)
        dicti_mann_whitney_traj["latitude " + var][ws][model1][model2] = (ulat, plat)
        dicti_mann_whitney_traj["latitude " + var][ws][model2][model1] = (ulat, plat)

    plot_dict(start_name, dicti_mann_whitney_traj, "traj_long", ["longitude no abs"])
    plot_dict(start_name, dicti_mann_whitney_traj, "traj_lat", ["latitude no abs"])
    plot_dict(start_name, dicti_mann_whitney_traj, "traj_speed_direction_long", ["longitude speed actual dir"])
    plot_dict(start_name, dicti_mann_whitney_traj, "traj_speed_direction_lat", ["latitude speed actual dir"])
    plot_dict(start_name, dicti_mann_whitney_traj, "traj_long_lat", ["longitude no abs", "latitude no abs"])
    plot_dict(start_name, dicti_mann_whitney_traj, "traj_speed_direction", ["longitude speed actual dir", "latitude speed actual dir"])