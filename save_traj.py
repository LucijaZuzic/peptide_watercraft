import os
import pandas as pd
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import numpy as np
from haversine import haversine

def haversine_array(x1, y1, x2, y2):
    x1corr = [max(-180, min(180, x)) for x in x1]
    y1corr = [max(-90, min(90, y)) for y in y1]
    x2corr = [max(-180, min(180, x)) for x in x2]
    y2corr = [max(-90, min(90, y)) for y in y2]
    return sum([haversine((y1corr[ix], x1corr[ix]), (y2corr[ix], x2corr[ix])) for ix in range(len(x1))]) / len(x1)
 
def euclidean(longitudes1, latitudes1, longitudes2, latitudes2):
    sum_dist = 0
    for i in range(len(longitudes1)):
        sum_dist += np.sqrt((longitudes1[i] - longitudes2[i]) ** 2 + (latitudes1[i] - latitudes2[i]) ** 2)
    return sum_dist / len(longitudes1)

sf1, sf2 = 5, 5

var_list = ["no abs", "speed actual dir"]
model_list = ["Bi", "Conv"]
ws_list = [2, 3, 4, 5, 10, 20, 30]

if not os.path.isfile("data_frame_traj_val_merged.csv"):

    data_frame_traj_val_new  = {"variable": [], "model": [], "ws": [], "test": [], "val": [], "R2": [], "MAE": [], "MSE": [], "RMSE": [], "euclid": [], "haversine": []}
    
    pd_file = pd.read_csv("data_frame_traj_val_old.csv", index_col = False)

    data_frame_traj_val_merged = dict()
    data_frame_traj_val_merged["variable"] = list(pd_file["variable"])
    data_frame_traj_val_merged["model"] = list(pd_file["model"])
    data_frame_traj_val_merged["ws"] = list(pd_file["ws"])
    data_frame_traj_val_merged["test"] = list(pd_file["test"])
    data_frame_traj_val_merged["val"] = list(pd_file["val"])
    data_frame_traj_val_merged["R2"] = list(pd_file["R2"])
    data_frame_traj_val_merged["MAE"] = list(pd_file["MAE"])
    data_frame_traj_val_merged["MSE"] = list(pd_file["MSE"])
    data_frame_traj_val_merged["RMSE"] = list(pd_file["RMSE"])
    data_frame_traj_val_merged["euclid"] = list(pd_file["euclid"])
    data_frame_traj_val_merged["haversine"] = list(pd_file["haversine"])
    
    vehicle_zero = os.listdir("csv_results_traj/1/1/")[0]
    ride_zero = os.listdir("csv_results_traj/1/1/" + vehicle_zero)[0]
    var_list = os.listdir("csv_results_traj/1/1/" + vehicle_zero + "/" + ride_zero + "/")
    model_list = os.listdir("csv_results_traj/1/1/" + vehicle_zero + "/" + ride_zero + "/" + var_list[0] + "/")
    for var in var_list:
        for model in model_list:
            for ws in ws_list:
                for nf1 in range(sf1):
                    for nf2 in range(sf2):
                        print(var, model, ws, nf1, nf2)
                        predicted_long = []
                        actual_long = []
                        predicted_lat = []
                        actual_lat = []
                        for vehicle in os.listdir("csv_results_traj/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"):
                            for ride in os.listdir("csv_results_traj/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + vehicle + "/"): 
                                path_to_file = "csv_results_traj/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + vehicle + "/" + str(ride) + "/" + var + "/" + model + "/" + str(ws) + "_predictions.csv"
                                pd_file = pd.read_csv(path_to_file, index_col = False)
                                predicted_long.extend(pd_file["predicted_long"])
                                actual_long.extend(pd_file["actual_long"])
                                predicted_lat.extend(pd_file["predicted_lat"])
                                actual_lat.extend(pd_file["actual_lat"])
                        actual = [[actual_long[ix], actual_lat[ix]] for ix in range(len(actual_long))]
                        predicted = [[predicted_long[ix], predicted_lat[ix]] for ix in range(len(predicted_long))]
                        euclidean_dist = euclidean(actual_long, actual_lat, predicted_long, predicted_lat)
                        haversine_dist = haversine_array(actual_long, actual_lat, predicted_long, predicted_lat)
                        r2_pred = r2_score(actual, predicted)
                        mae_pred = mean_absolute_error(actual, predicted)
                        mse_pred = mean_squared_error(actual, predicted)
                        rmse_pred = math.sqrt(mse_pred)
                        print(r2_pred, mae_pred, mse_pred, rmse_pred, euclidean_dist, haversine_dist)
                        data_frame_traj_val_new["variable"].append(var)
                        data_frame_traj_val_new["model"].append(model)
                        data_frame_traj_val_new["ws"].append(ws)
                        data_frame_traj_val_new["test"].append(nf1 + 1)
                        data_frame_traj_val_new["val"].append(nf2 + 1)
                        data_frame_traj_val_new["R2"].append(r2_pred)
                        data_frame_traj_val_new["MAE"].append(mae_pred)
                        data_frame_traj_val_new["MSE"].append(mse_pred)
                        data_frame_traj_val_new["RMSE"].append(rmse_pred)
                        data_frame_traj_val_new["euclid"].append(euclidean_dist)
                        data_frame_traj_val_new["haversine"].append(haversine_dist)
                        data_frame_traj_val_merged["variable"].append(var)
                        data_frame_traj_val_merged["model"].append(model)
                        data_frame_traj_val_merged["ws"].append(ws)
                        data_frame_traj_val_merged["test"].append(nf1 + 1)
                        data_frame_traj_val_merged["val"].append(nf2 + 1)
                        data_frame_traj_val_merged["R2"].append(r2_pred)
                        data_frame_traj_val_merged["MAE"].append(mae_pred)
                        data_frame_traj_val_merged["MSE"].append(mse_pred)
                        data_frame_traj_val_merged["RMSE"].append(rmse_pred)
                        data_frame_traj_val_merged["euclid"].append(euclidean_dist)
                        data_frame_traj_val_merged["haversine"].append(haversine_dist)

    df_data_frame_traj_val_new = pd.DataFrame(data_frame_traj_val_new)
    df_data_frame_traj_val_new.to_csv("data_frame_traj_val_new.csv", index = False)
    
    df_data_frame_traj_val_merged = pd.DataFrame(data_frame_traj_val_merged)
    df_data_frame_traj_val_merged.to_csv("data_frame_traj_val_merged.csv", index = False)