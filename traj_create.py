import os
import pandas as pd
import pickle
import numpy as np

def load_object(file_name): 
    with open(file_name, 'rb') as file_object:
        data = pickle.load(file_object) 
        file_object.close()
        return data
    
def get_sides_from_angle(longest, angle):
    return longest * np.cos(angle / 180 * np.pi), longest * np.sin(angle / 180 * np.pi)

dictangle = dict()
def change_angle(angle, name_file):
        
    if name_file not in dictangle:
        file_with_ride = pd.read_csv(name_file) 
        x_dir = list(file_with_ride["fields_longitude"])[0] < list(file_with_ride["fields_longitude"])[-1]
        y_dir = list(file_with_ride["fields_latitude"])[0] < list(file_with_ride["fields_latitude"])[-1]
        dictangle[name_file] = (x_dir, y_dir)
    else:
        x_dir, y_dir = dictangle[name_file]
        
    new_dir = (90 - angle + 360) % 360 
    if not x_dir: 
        new_dir = (180 - new_dir + 360) % 360
    if not y_dir: 
        new_dir = 360 - new_dir 

    return new_dir

sf1, sf2 = 5, 5

vehicle_zero = os.listdir("csv_results/1/1/")[0]
ride_zero = os.listdir("csv_results/1/1/" + vehicle_zero)[0]
var_list = os.listdir("csv_results/1/1/" + vehicle_zero + "/" + ride_zero + "/")
model_list = os.listdir("csv_results/1/1/" + vehicle_zero + "/" + ride_zero + "/" + var_list[0] + "/")

for nf2 in range(sf2):
    for nf1 in range(sf1):
        for model in model_list:
            for ws in [2, 3, 4, 5, 10, 20, 30]:
                for vehicle in os.listdir("csv_results/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"):
                    for ride in os.listdir("csv_results/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + vehicle + "/"):
                        kfile = vehicle + "/cleaned_csv/" + ride + ".csv"
                        pd_file_dict = dict()
                        for var in var_list:
                            path_to_file = "csv_results/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + vehicle + "/" + str(ride) + "/" + var + "/" + model + "/" + str(ws) + "_predictions.csv"
                            pd_file_dict[var] = pd.read_csv(path_to_file, index_col = False)
                       
                        actual_trajs = {"no abs": {"predicted_long": [0], "actual_long": [0], "predicted_lat": [0], "actual_lat": [0]}, 
                                        "speed actual dir": {"predicted_long": [0], "actual_long": [0], "predicted_lat": [0], "actual_lat": [0]}, 
                                        "speed ones dir": {"predicted_long": [0], "actual_long": [0], "predicted_lat": [0], "actual_lat": [0]}}
                        
                        actual = dict()
                        for var in var_list:
                            actual[var] = load_object("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_" + var)[kfile]
                        
                        actual["time"] = load_object("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_time")[kfile]
                        
                        for ix in range(len(pd_file_dict["longitude_no_abs"]["predicted_descaled"])):
                            actual_trajs["no abs"]["predicted_long"].append(actual_trajs["no abs"]["predicted_long"][-1] + pd_file_dict["longitude_no_abs"]["predicted_descaled"][ix])
                            actual_trajs["no abs"]["predicted_lat"].append(actual_trajs["no abs"]["predicted_lat"][-1] + pd_file_dict["latitude_no_abs"]["predicted_descaled"][ix])
                            actual_trajs["no abs"]["actual_long"].append(actual_trajs["no abs"]["actual_long"][-1] + actual["longitude_no_abs"][ix])
                            actual_trajs["no abs"]["actual_lat"].append(actual_trajs["no abs"]["actual_lat"][-1] + actual["latitude_no_abs"][ix])
                        
                        for ix in range(len(pd_file_dict["speed"]["predicted_descaled"])):
                            speed_pred = pd_file_dict["speed"]["predicted_descaled"][ix]
                            dir_pred = pd_file_dict["direction"]["predicted_descaled"][ix]
                            speed_new = speed_pred / 111 / 0.1 / 3600
                            dir_new = change_angle(dir_pred, kfile)
                            new_long_time, new_lat_time = get_sides_from_angle(actual["time"][ix] * speed_new, dir_new)
                            new_long_ones, new_lat_ones = get_sides_from_angle(speed_new, dir_new)

                            actual_trajs["speed actual dir"]["predicted_long"].append(actual_trajs["speed actual dir"]["predicted_long"][-1] + new_long_time)
                            actual_trajs["speed actual dir"]["predicted_lat"].append(actual_trajs["speed actual dir"]["predicted_lat"][-1] + new_lat_time)
                            actual_trajs["speed actual dir"]["actual_long"].append(actual_trajs["speed actual dir"]["actual_long"][-1] + actual["longitude_no_abs"][ix])
                            actual_trajs["speed actual dir"]["actual_lat"].append(actual_trajs["speed actual dir"]["actual_lat"][-1] + actual["latitude_no_abs"][ix])
                            continue
                            actual_trajs["speed ones dir"]["predicted_long"].append(actual_trajs["speed ones dir"]["predicted_long"][-1] + new_long_ones)
                            actual_trajs["speed ones dir"]["predicted_lat"].append(actual_trajs["speed ones dir"]["predicted_lat"][-1] + new_lat_ones)
                            actual_trajs["speed ones dir"]["actual_long"].append(actual_trajs["speed ones dir"]["actual_long"][-1] + actual["longitude_no_abs"][ix])
                            actual_trajs["speed ones dir"]["actual_lat"].append(actual_trajs["speed ones dir"]["actual_lat"][-1] + actual["latitude_no_abs"][ix])
                            
                        for m in actual_trajs:
                            if "ones" in m:
                                continue

                            print(m, len(actual_trajs[m]["predicted_long"]))

                            path_to_file = "csv_results_traj/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + vehicle + "/" + str(ride) + "/" + m + "/" + model + "/"
                            df_save = pd.DataFrame(actual_trajs[m])
                            if not os.path.isdir(path_to_file):
                                os.makedirs(path_to_file)
                            df_save.to_csv(path_to_file + str(ws) + "_predictions.csv", index = False)
                            print(path_to_file + str(ws) + "_predictions.csv")