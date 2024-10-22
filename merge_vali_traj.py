import os
import pandas as pd

sf1, sf2 = 5, 5

vehicle_zero = os.listdir("csv_results_traj/1/1/")[0]
ride_zero = os.listdir("csv_results_traj/1/1/" + vehicle_zero)[0]
var_list = os.listdir("csv_results_traj/1/1/" + vehicle_zero + "/" + ride_zero + "/")
model_list = os.listdir("csv_results_traj/1/1/" + vehicle_zero + "/" + ride_zero + "/" + var_list[0] + "/")
for var in var_list:
    for ws in [2, 3, 4, 5, 10, 20, 30]:
        dict_pred = dict()
        for nf2 in range(sf2):
            for nf1 in range(sf1):
                for vehicle in os.listdir("csv_results_traj/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"):
                    for ride in os.listdir("csv_results_traj/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + vehicle + "/"): 
                        for model in model_list:
                            path_to_file = "csv_results_traj/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + vehicle + "/" + str(ride) + "/" + var + "/" + model + "/" + str(ws) + "_predictions.csv"
                            pd_file = pd.read_csv(path_to_file, index_col = False)
                            if model + " long" not in dict_pred:
                                dict_pred[model + " long"] = []
                            dict_pred[model + " long"].extend(pd_file["predicted_long"])
                            if model + " lat" not in dict_pred:
                                dict_pred[model + " lat"] = []
                            dict_pred[model + " lat"].extend(pd_file["predicted_lat"])
                            if model == model_list[0]:
                                if "actual long" not in dict_pred:
                                    dict_pred["actual long"] = []
                                dict_pred["actual long"].extend(pd_file["actual_long"])
                                if "actual lat" not in dict_pred:
                                    dict_pred["actual lat"] = []
                                dict_pred["actual lat"].extend(pd_file["actual_lat"])
        ride_name = "csv_results_traj_merged_validation/total/" + var + "/"
        if not os.path.isdir(ride_name):
            os.makedirs(ride_name)
        df_write = pd.DataFrame(dict_pred)
        df_write.to_csv(ride_name + str(ws) + "_predictions.csv", index = False)
        print(ride_name + str(ws) + "_predictions.csv")