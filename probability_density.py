import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import pickle

def load_object(file_name): 
    with open(file_name, 'rb') as file_object:
        data = pickle.load(file_object) 
        file_object.close()
        return data

cm = 1/2.54  # centimeters in inches

def set_params():
    
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

def make_hist(list_lens, title_use):
    print(title_use)
    print(np.sum(list_lens))
    print(min(list_lens), max(list_lens), np.average(list_lens), np.mean(list_lens), np.median(list_lens))
    print(np.percentile(list_lens, 0), np.percentile(list_lens, 25), np.percentile(list_lens, 50), np.percentile(list_lens, 75), np.percentile(list_lens, 100))
    plt.figure(figsize=(29.7/4*cm, 29.7/4*cm), dpi = 300)
    set_params()
    varnew = title_use.replace("_", " ").replace("longitude no abs", "$x$ offset").replace("direction", "heading")
    varnew = varnew.replace("latitude no abs", "$y$ offset").replace("Area x", "Area $x$").replace("Area y", "Area $y$")
    plt.hist(list_lens, color = "#004488", bins = 100)
    if "time" == title_use:
        plt.xlabel(varnew.capitalize() + " ($s$)")
    if "duration" == title_use:
        plt.xlabel(varnew.capitalize() + " ($s$)")
    if "speed" == title_use:
        plt.xlabel(varnew.capitalize() + " ($km/h$)")
    if "longitude_no_abs" == title_use:
        plt.xlabel(varnew.capitalize() + " ($\degree$)")
    if "latitude_no_abs" == title_use:
        plt.xlabel(varnew.capitalize() + " ($\degree$)")
    if "direction" == title_use:
        plt.xlabel(varnew.capitalize() + " ($\degree$)")
    if "Area" in title_use and "total" not in title_use:
        plt.xlabel(varnew.capitalize() + " ($\degree$)")
    if "Area" in title_use and "total" in title_use:
        plt.xlabel(varnew.capitalize() + " ($\degree^{2}$)")
    if "Size" in title_use and "total" not in title_use:
        plt.xlabel("Number of points")
    plt.ylabel("Frequency")
    if not os.path.isdir("hist_plot"):
        os.makedirs("hist_plot")
    plt.savefig("hist_plot/" + title_use.lower().replace(" ", "_") + ".png", bbox_inches = "tight")
    plt.savefig("hist_plot/" + title_use.lower().replace(" ", "_") + ".svg", bbox_inches = "tight")
    plt.savefig("hist_plot/" + title_use.lower().replace(" ", "_") + ".pdf", bbox_inches = "tight")
    plt.close()

sf1, sf2 = 5, 3

num_to_ws = [-1, 5, 6, 5, 6, 5, 6, 5, 6, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 2, 10, 20, 30, 3, 3, 3, 3, 4, 4, 4, 4, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 15, 15, 15, 15, 19, 19, 19, 19, 25, 25, 25, 25, 29, 29, 29, 29, 16, 16, 16, 16, 32, 32, 32, 32]

num_to_params = [-1, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]

actual_var = dict()
actual_var_all = dict()
lens_dict = dict()
for nf2 in [sf2]:
    for nf1 in range(sf1):
        ride_name = "csv_results/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"
        varnames = os.listdir("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/")
        for v in varnames:
            var = v.replace("actual_", "")
            actual_var_par = load_object("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + v)
            for r in actual_var_par:
                if r not in actual_var:
                    actual_var[r] = dict()
                actual_var[r][var] = actual_var_par[r]
                if var not in actual_var_all:
                    actual_var_all[var] = []
                actual_var_all[var].extend(actual_var_par[r])
                if var not in lens_dict:
                    lens_dict[var] = dict()
                lens_dict[var][r] = len(actual_var_par[r])
print(len(actual_var))
print(list(actual_var_all.keys()))
for var in lens_dict:
    make_hist(list(lens_dict[var].values()), "Size " + var)
    print(sum(list(lens_dict[var].values())))
for var in actual_var_all:
    make_hist(actual_var_all[var], var)
ix_plot = 0
plt.figure(figsize=(21*cm, 29.7/1.8*cm), dpi = 300)
set_params()
for var in actual_var_all:
    if "time" in var:
        continue
    ix_plot += 1
    plt.subplot(2, 2, ix_plot)
    varnew = var.replace("_", " ").replace("longitude no abs", "$x$ offset").replace("direction", "heading")
    varnew = varnew.replace("latitude no abs", "$y$ offset")
    if "speed" == var:
        plt.xlabel(varnew.capitalize() + " ($km/h$)")
    if "longitude_no_abs" == var:
        plt.xlabel(varnew.capitalize() + " ($\degree$)")
    if "latitude_no_abs" == var:
        plt.xlabel(varnew.capitalize() + " ($\degree$)")
    if "direction" == var:
        plt.xlabel(varnew.capitalize() + " ($\degree$)")
    plt.ylabel("Frequency")
    plt.hist(actual_var_all[var], color = "#004488", bins = 100)
    if not os.path.isdir("hist_plot"):
        os.makedirs("hist_plot")
plt.savefig("hist_plot/all_var_no_time_hist.png", bbox_inches = "tight")
plt.savefig("hist_plot/all_var_no_time_hist.svg", bbox_inches = "tight")
plt.savefig("hist_plot/all_var_no_time_hist.pdf", bbox_inches = "tight")
plt.close()
ix_plot = 0
plt.figure(figsize=(21*cm, 29.7/1.4*cm), dpi = 300)
set_params()
for var in actual_var_all:
    ix_plot += 1
    plt.subplot(3, 2, ix_plot)
    varnew = var.replace("_", " ").replace("longitude no abs", "$x$ offset").replace("direction", "heading")
    varnew = varnew.replace("latitude no abs", "$y$ offset")
    if "time" == var:
        plt.xlabel(varnew.capitalize() + " ($s$)")
    if "speed" == var:
        plt.xlabel(varnew.capitalize() + " ($km/h$)")
    if "longitude_no_abs" == var:
        plt.xlabel(varnew.capitalize() + " ($\degree$)")
    if "latitude_no_abs" == var:
        plt.xlabel(varnew.capitalize() + " ($\degree$)")
    if "direction" == var:
        plt.xlabel(varnew.capitalize() + " ($\degree$)")
    plt.ylabel("Frequency")
    plt.hist(actual_var_all[var], color = "#004488", bins = 100)
    if not os.path.isdir("hist_plot"):
        os.makedirs("hist_plot")
plt.savefig("hist_plot/all_var_hist.png", bbox_inches = "tight")
plt.savefig("hist_plot/all_var_hist.svg", bbox_inches = "tight")
plt.savefig("hist_plot/all_var_hist.pdf", bbox_inches = "tight")
plt.close()
area_x_dict = dict()
area_y_dict = dict()
area_total_dict = dict()
duration_total_dict = dict()
for r in actual_var:
    list_long = actual_var[r]["longitude_no_abs"]
    cumulative_long = list(np.cumsum(list_long))
    cumulative_long.insert(0, 0)
    actual_var[r]["longitude_cumulative"] = cumulative_long
    list_lat = actual_var[r]["latitude_no_abs"]
    cumulative_lat = list(np.cumsum(list_lat))
    cumulative_lat.insert(0, 0)
    actual_var[r]["latitude_cumulative"] = cumulative_lat
    area_x_dict[r] = max(cumulative_long) - min(cumulative_long)
    area_y_dict[r] = max(cumulative_lat) - min(cumulative_lat)
    area_total_dict[r] = area_x_dict[r] * area_y_dict[r]
    list_time = actual_var[r]["time"]
    duration_total_dict[r] = np.sum(list_time)
list_lens = list(area_x_dict.values())
make_hist(list(area_x_dict.values()), "Area x")
make_hist(list(area_y_dict.values()), "Area y")
make_hist(list(area_total_dict.values()), "Area total")
make_hist(list(duration_total_dict.values()), "duration")
plt.figure(figsize=(21*cm, 29.7/1.8*cm), dpi = 300)
set_params()
plt.subplot(2, 2, 1)
plt.xlabel("Number of points")
plt.ylabel("Frequency")
plt.hist(list(lens_dict["speed"].values()), color = "#004488", bins = 100)
plt.subplot(2, 2, 2)
plt.xlabel("Longitude range ($\degree$)")
plt.ylabel("Frequency")
plt.hist(list(area_x_dict.values()), color = "#004488", bins = 100)
plt.subplot(2, 2, 3)
plt.xlabel("Latitude range ($\degree$)")
plt.ylabel("Frequency")
plt.hist(list(area_y_dict.values()), color = "#004488", bins = 100)
plt.subplot(2, 2, 4)
plt.xlabel("Total area ($\degree^{2}$)")
plt.ylabel("Frequency")
plt.hist(list(area_total_dict.values()), color = "#004488", bins = 100)
if not os.path.isdir("hist_plot"):
    os.makedirs("hist_plot")
plt.savefig("hist_plot/all_traj_features.png", bbox_inches = "tight")
plt.savefig("hist_plot/all_traj_features.svg", bbox_inches = "tight")
plt.savefig("hist_plot/all_traj_features.pdf", bbox_inches = "tight")
plt.close()
plt.figure(figsize=(21*cm, 29.7/1.4*cm), dpi = 300)
set_params()
plt.subplot(3, 2, 1)
plt.xlabel("Number of points")
plt.ylabel("Frequency")
plt.hist(list(lens_dict["speed"].values()), color = "#004488", bins = 100)
plt.subplot(3, 2, 2)
plt.xlabel("Duration ($s$)")
plt.ylabel("Frequency")
plt.hist(list(duration_total_dict.values()), color = "#004488", bins = 100)
plt.subplot(3, 2, 3)
plt.xlabel("Longitude range ($\degree$)")
plt.ylabel("Frequency")
plt.hist(list(area_x_dict.values()), color = "#004488", bins = 100)
plt.subplot(3, 2, 4)
plt.xlabel("Latitude range ($\degree$)")
plt.ylabel("Frequency")
plt.hist(list(area_y_dict.values()), color = "#004488", bins = 100)
plt.subplot(3, 2, 5)
plt.xlabel("Total area ($\degree^{2}$)")
plt.ylabel("Frequency")
plt.hist(list(area_total_dict.values()), color = "#004488", bins = 100)
if not os.path.isdir("hist_plot"):
    os.makedirs("hist_plot")
plt.savefig("hist_plot/all_traj_features_time.png", bbox_inches = "tight")
plt.savefig("hist_plot/all_traj_features_time.svg", bbox_inches = "tight")
plt.savefig("hist_plot/all_traj_features_time.pdf", bbox_inches = "tight")
plt.close()
ix_plot = 0
plt.figure(figsize = (10, 10 * 21 / 19), dpi = 300)
set_params()
for r in actual_var:
    ix_plot += 1
    plt.subplot(21, 19, ix_plot)
    plt.rcParams.update({'font.size': 28}) 
    plt.rcParams['font.family'] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.axis("equal")
    plt.axis("off")
    plt.plot(actual_var[r]["longitude_cumulative"], actual_var[r]["latitude_cumulative"], c = "k", linewidth = 1)
plt.savefig("hist_plot/all_trajs.png", bbox_inches = "tight")
plt.savefig("hist_plot/all_trajs.svg", bbox_inches = "tight")
plt.savefig("hist_plot/all_trajs.pdf", bbox_inches = "tight")
plt.close()
redo_res = False
redo_all = False
if redo_res:
    read_UniTS_4 = dict()

    vehicle_zero = os.listdir("csv_results_traj/1/1/")[0]
    ride_zero = os.listdir("csv_results_traj/1/1/" + vehicle_zero)[0]
    var_list = os.listdir("csv_results_traj/1/1/" + vehicle_zero + "/" + ride_zero + "/")
    model_list = os.listdir("csv_results_traj/1/1/" + vehicle_zero + "/" + ride_zero + "/" + var_list[0] + "/")
    ws_long_list = os.listdir("csv_results_traj/1/1/" + vehicle_zero + "/" + ride_zero + "/" + var_list[0] + "/" + model_list[0] + "/")

    for var in ["speed actual dir"]:
        for model in ["UniTS"]:
            for ws_long in ["2_predictions.csv", "30_predictions.csv"]:
                for nf2 in [sf2]:
                    actual_rides = dict()
                    predicted_rides = dict()
                    for nf1 in range(sf1):
                        for vehicle in os.listdir("csv_results_traj/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"):
                            for ride in os.listdir("csv_results_traj/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + vehicle):
                                traj_path = "csv_results_traj/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + vehicle  + "/" + ride + "/" + var + "/" + model + "/" + ws_long
                                pd_file = pd.read_csv(traj_path, index_col = False)
                                predicted_rides[vehicle + "/cleaned_csv/" + ride + ".csv"] = dict()
                                actual_rides[vehicle + "/cleaned_csv/" + ride + ".csv"] = dict()
                                predicted_rides[vehicle + "/cleaned_csv/" + ride + ".csv"]["long"] = pd_file["predicted long"]
                                actual_rides[vehicle + "/cleaned_csv/" + ride + ".csv"]["long"] = pd_file["actual long"]
                                predicted_rides[vehicle + "/cleaned_csv/" + ride + ".csv"]["lat"] = pd_file["predicted lat"]
                                actual_rides[vehicle + "/cleaned_csv/" + ride + ".csv"]["lat"] = pd_file["actual lat"]
                    ix_plot = 0
                    plt.figure(figsize = (10, 10 * 21 / 19), dpi = 300)
                    set_params()
                    for r in actual_var:
                        ix_plot += 1
                        plt.subplot(21, 19, ix_plot)
                        plt.rcParams.update({'font.size': 28}) 
                        plt.rcParams['font.family'] = "serif"
                        plt.rcParams["mathtext.fontset"] = "dejavuserif"
                        plt.axis("equal")
                        plt.axis("off")
                        plt.plot(actual_rides[r]["long"], actual_rides[r]["lat"], c = "k", linewidth = 1)
                        plt.plot(predicted_rides[r]["long"], predicted_rides[r]["lat"], c  = "#004488", linewidth = 1)
                    
                    filepath = "hist_plot/" + var + "/" + model + "/" + str(nf2 + 1) + "/"
                    if not os.path.isdir(filepath):
                        os.makedirs(filepath)
                    filename = ws_long.replace("_predictions.csv", "_merge_val")
                    plt.savefig(filepath + filename + ".png", bbox_inches = "tight")
                    plt.savefig(filepath + filename + ".svg", bbox_inches = "tight")
                    plt.savefig(filepath + filename + ".pdf", bbox_inches = "tight")
                    plt.close()
                    if not redo_all or ws_long != "2_predictions.csv" or model != "UniTS" or var != "no abs":
                        continue
                    ix_plot = 0
                    plt.figure(figsize = (10, 10 * 21 / 19), dpi = 300)
                    set_params()
                    for r in actual_var:
                        ix_plot += 1
                        plt.subplot(21, 19, ix_plot)
                        plt.rcParams.update({'font.size': 28}) 
                        plt.rcParams['font.family'] = "serif"
                        plt.rcParams["mathtext.fontset"] = "dejavuserif"
                        plt.axis("equal")
                        plt.axis("off")
                        plt.plot(actual_rides[r]["long"], actual_rides[r]["lat"], c = "k", linewidth = 1)
                    plt.savefig("hist_plot/all_trajs_compared.png", bbox_inches = "tight")
                    plt.savefig("hist_plot/all_trajs_compared.svg", bbox_inches = "tight")
                    plt.savefig("hist_plot/all_trajs_compared.pdf", bbox_inches = "tight")
                    plt.close()