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

sf1, sf2 = 5, 4

actual_var = dict()
actual_var_all = dict()
lens_dict = dict()
for nf1 in range(sf1):
    for nf2 in [sf2]:
        ride_name = "csv_results/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"
        varnames = os.listdir("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/")
        for v in varnames:
            var = v.replace("actual_", "")
            actual_var_par = load_object("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + v)
            for r in actual_var_par:
                if r not in actual_var:
                    actual_var[r] = dict()
                actual_var[r][var] = actual_var_par[r]
print(actual_var.keys())
included = [
    "1_4695799",
    "1_4718376",
    "2_9309752",
    "2_9485205",
    "3_9018208",
    "3_9149814",
    "4_9381297",
    "4_9471375",
    "6_8364490",
    "6_9433629",
    "8_8366414",
    "8_9151191",
    "9_8712338",
    "9_8892048",
    "10_9014243",
    "10_9039658",
    "11_8604891",
    "11_9003337",
    "12_8478762",
    "12_8804925",
    "13_8521037",
    "13_8569512",
    "15_9114544",
    "15_9151549",
    "16_8972258",
    "16_9206091",
    "17_8597316",
    "17_9142047"
]

ix_plot = 0
plt.figure(figsize = (10, 10 * 4 / 7), dpi = 300)
set_params()

for name in included:
    vehicle_split = name.split("_")
    vehicle = vehicle_split[0]
    ride = vehicle_split[1]
    r = "Vehicle_" + str(vehicle) + "/cleaned_csv/events_" + str(ride) + ".csv"
    pd_file = pd.read_csv(r, index_col = False)
    list_long = pd_file["fields_longitude"]
    list_lat = pd_file["fields_latitude"]
    ix_plot += 1
    plt.subplot(4, 7, ix_plot)
    plt.rcParams.update({'font.size': 28}) 
    plt.rcParams['font.family'] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.axis("equal")
    plt.axis("off")
    plt.plot(list_long, list_lat, c = "k", linewidth = 0.5)

plt.savefig("baseline.png", bbox_inches = "tight")
plt.savefig("baseline.svg", bbox_inches = "tight")
plt.savefig("baseline.pdf", bbox_inches = "tight")
plt.close()