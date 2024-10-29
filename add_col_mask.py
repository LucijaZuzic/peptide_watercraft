import os
import pandas as pd
import pickle
import numpy as np

def load_object(file_name): 
    with open(file_name, 'rb') as file_object:
        data = pickle.load(file_object) 
        file_object.close()
        return data
 
def get_XY(dat, time_steps, len_skip = -1, len_output = -1, max_len = 30, mask_value = 2):
    X = []
    Y = [] 
    if len_skip == -1:
        len_skip = time_steps
    if len_output == -1:
        len_output = time_steps
    for i in range(0, len(dat), len_skip):
        x_vals = dat[i:min(i + time_steps, len(dat))]
        y_vals = dat[i + time_steps:i + time_steps + len_output]
        if len(x_vals) == time_steps and len(y_vals) == len_output:
            while len(x_vals) < max_len:
                x_vals.append(mask_value)
            X.append(np.array(x_vals))
            while len(y_vals) < max_len:
                y_vals.append(mask_value)
            Y.append(np.array(y_vals))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y
    
sf1, sf2 = 5, 5

for nf1 in range(sf1):
    for nf2 in range(sf2):

        ride_name = "csv_results_mask/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"

        varnames = os.listdir("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/")
        varnames = [v.replace("actual_", "") for v in varnames]
            
        x_test_dict = dict()
        y_test_dict = dict()
        for varname in varnames:
            if "time" in varname:
                continue
            x_test_dict[varname] = dict()
            y_test_dict[varname] = dict()
            for ws in [2, 3, 4, 5, 10, 20, 30]:
                x_test_dict[varname][ws] = dict()
                y_test_dict[varname][ws] = dict()
                loaded_obj = load_object("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_" + varname)
                for ride in loaded_obj:
                    x_test_dict[varname][ws][ride] = []
                    y_test_dict[varname][ws][ride] = []
                    x_test_part, y_test_part = get_XY(loaded_obj[ride], ws, ws, ws, ws, 2)
                    for ix in range(len(x_test_part)):
                        x_test_dict[varname][ws][ride].extend(x_test_part[ix]) 
                        y_test_dict[varname][ws][ride].extend(y_test_part[ix])

        len_dict = dict()
        for ws in [2, 3, 4, 5, 10, 20, 30]:
            len_dict[ws] = dict()
            for ride in y_test_dict["speed"][ws]:
                len_dict[ws][ride] = 1000000
                for varname in varnames:
                    if "time" in varname:
                        continue
                    len_dict[ws][ride] = min(len_dict[ws][ride], len(y_test_dict[varname][ws][ride]))

        for varname in varnames:
            if "time" in varname:
                continue
            if "abs" not in varname:
                for model_name in ["Bi_Mask", "Conv_Mask"]:
                    for ws in [2, 3, 4, 5, 10, 20, 30]:
                            saving_dir = model_name + "/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + str(ws)
                            saving_name = saving_dir + "/" + saving_dir.replace("/", "_") + "_" + varname
                            pd_file = pd.read_csv(saving_name + "_test_pred.csv")
                            all_actual_actual = []
                            so_far = 0

                            for ride in y_test_dict[varname][ws]:

                                x_test_all_short = x_test_dict[varname][ws][ride][:len_dict[ws][ride]]
                                y_test_all_short = y_test_dict[varname][ws][ride][:len_dict[ws][ride]]
                                
                                dict_write = dict()

                                dict_write = {"predicted": list(pd_file["predicted"][so_far:so_far+len(y_test_all_short)]),
                                            "predicted_descaled": list(pd_file["predicted_descaled"][so_far:so_far+len(y_test_all_short)]),
                                            "actual": list(pd_file["actual"][so_far:so_far+len(y_test_all_short)]),
                                            "actual_descaled": list(pd_file["actual_descaled"][so_far:so_far+len(y_test_all_short)]),
                                            "actual_actual": y_test_all_short}
                                
                                so_far += len(y_test_all_short)

                                all_actual_actual.extend(y_test_all_short)

                                ride_name = "csv_results_mask/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + ride.replace(".csv", "/").replace("/cleaned_csv/", "/") + str(varname) + "/" + model_name + "/" 
                                if not os.path.isdir(ride_name):
                                    os.makedirs(ride_name)
                                df_write = pd.DataFrame(dict_write)
                                df_write.to_csv(ride_name + str(ws) + "_predictions.csv", index = False)
                                print(ride_name + str(ws) + "_predictions.csv")
                            
                            dict_write_all = dict()

                            dict_write_all = {"predicted": list(pd_file["predicted"]),
                                            "predicted_descaled": list(pd_file["predicted_descaled"]),
                                            "actual": list(pd_file["actual"]),
                                            "actual_descaled": list(pd_file["actual_descaled"]),
                                            "actual_actual": all_actual_actual}

                            df_pd_file = pd.DataFrame(dict_write_all)
                            df_pd_file.to_csv(saving_name + "_test_pred_with_actual.csv", index = False)
            else:
                for model_name in ["Bi_Mask", "Conv_Mask"]:
                    for ws in [2, 3, 4, 5, 10, 20, 30]:
                            saving_dir = model_name + "/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + str(ws)
                            saving_name = saving_dir + "/" + saving_dir.replace("/", "_") + "_" + varname
                            pd_file_abs = pd.read_csv(saving_name + "_abs_test_pred.csv")
                            pd_file_sgn = pd.read_csv(saving_name + "_sgn_test_pred.csv")
                            all_actual_actual = []
                            so_far = 0

                            for ride in y_test_dict[varname][ws]:

                                x_test_all_short = x_test_dict[varname][ws][ride][:len_dict[ws][ride]]
                                y_test_all_short = y_test_dict[varname][ws][ride][:len_dict[ws][ride]]

                                dict_write = dict()

                                predicted_abs = list(pd_file_abs["predicted"][so_far:so_far+len(y_test_all_short)])
                                predicted_sgn = list(pd_file_sgn["predicted"][so_far:so_far+len(y_test_all_short)])

                                predicted_descaled_abs = list(pd_file_abs["predicted_descaled"][so_far:so_far+len(y_test_all_short)])
                                predicted_descaled_sgn = list(pd_file_sgn["predicted_descaled"][so_far:so_far+len(y_test_all_short)])

                                actual_abs = list(pd_file_abs["actual"][so_far:so_far+len(y_test_all_short)])
                                actual_sgn = list(pd_file_sgn["actual"][so_far:so_far+len(y_test_all_short)])

                                actual_descaled_abs = list(pd_file_abs["actual_descaled"][so_far:so_far+len(y_test_all_short)])
                                actual_descaled_sgn = list(pd_file_sgn["actual_descaled"][so_far:so_far+len(y_test_all_short)])

                                predicted_descaled_sgn_int = [(int(predicted_descaled_sgn[ix_val] > 0.5) * 2 - 1) for ix_val in range(len(predicted_descaled_sgn))]
                                actual_descaled_sgn_int = [(int(actual_descaled_sgn[ix_val] > 0.5) * 2 - 1) for ix_val in range(len(actual_descaled_sgn))]

                                predicted_total = [predicted_descaled_sgn_int[ix_val] * predicted_descaled_abs[ix_val] for ix_val in range(len(predicted_descaled_abs))]
                                actual_total = [actual_descaled_sgn_int[ix_val] * actual_descaled_abs[ix_val] for ix_val in range(len(actual_descaled_abs))]
                                
                                dict_write = {"predicted_abs": predicted_abs,
                                              "predicted_sgn": predicted_sgn,
                                              "predicted_descaled_abs": predicted_descaled_abs,
                                              "predicted_descaled_sgn": predicted_descaled_sgn,
                                              "actual_abs": actual_abs,
                                              "actual_sgn": actual_sgn,
                                              "actual_descaled_abs": actual_descaled_abs,
                                              "actual_descaled_sgn": actual_descaled_sgn,
                                              "predicted_descaled_sgn_int": predicted_descaled_sgn_int,
                                              "actual_descaled_sgn_int": actual_descaled_sgn_int,
                                              "predicted_total": predicted_total,
                                              "actual_total": actual_total,
                                              "actual_actual": y_test_all_short
                                              }
                                
                                so_far += len(y_test_all_short)

                                all_actual_actual.extend(y_test_all_short)

                                ride_name = "csv_results_mask/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + ride.replace(".csv", "/").replace("/cleaned_csv/", "/") + str(varname) + "/" + model_name + "/" 
                                if not os.path.isdir(ride_name):
                                    os.makedirs(ride_name)
                                df_write = pd.DataFrame(dict_write)
                                df_write.to_csv(ride_name + str(ws) + "_predictions.csv", index = False)
                                print(ride_name + str(ws) + "_predictions.csv")
                            
                            dict_write_all = dict()

                            predicted_abs = list(pd_file_abs["predicted"])
                            predicted_sgn = list(pd_file_sgn["predicted"])

                            predicted_descaled_abs = list(pd_file_abs["predicted_descaled"])
                            predicted_descaled_sgn = list(pd_file_sgn["predicted_descaled"])

                            actual_abs = list(pd_file_abs["actual"])
                            actual_sgn = list(pd_file_sgn["actual"])

                            actual_descaled_abs = list(pd_file_abs["actual_descaled"])
                            actual_descaled_sgn = list(pd_file_sgn["actual_descaled"])

                            predicted_descaled_sgn_int = [(int(predicted_descaled_sgn[ix_val] > 0.5) * 2 - 1) for ix_val in range(len(predicted_descaled_sgn))]
                            actual_descaled_sgn_int = [(int(actual_descaled_sgn[ix_val] > 0.5) * 2 - 1) for ix_val in range(len(actual_descaled_sgn))]

                            predicted_total = [predicted_descaled_sgn_int[ix_val] * predicted_descaled_abs[ix_val] for ix_val in range(len(predicted_descaled_abs))]
                            actual_total = [actual_descaled_sgn_int[ix_val] * actual_descaled_abs[ix_val] for ix_val in range(len(actual_descaled_abs))]
                            
                            dict_write_all = {"predicted_abs": predicted_abs,
                                            "predicted_sgn": predicted_sgn,
                                            "predicted_descaled_abs": predicted_descaled_abs,
                                            "predicted_descaled_sgn": predicted_descaled_sgn,
                                            "actual_abs": actual_abs,
                                            "actual_sgn": actual_sgn,
                                            "actual_descaled_abs": actual_descaled_abs,
                                            "actual_descaled_sgn": actual_descaled_sgn,
                                            "predicted_descaled_sgn_int": predicted_descaled_sgn_int,
                                            "actual_descaled_sgn_int": actual_descaled_sgn_int,
                                            "predicted_total": predicted_total,
                                            "actual_total": actual_total,
                                            "actual_actual": all_actual_actual
                                            }

                            df_pd_file = pd.DataFrame(dict_write_all)
                            df_pd_file.to_csv(saving_name + "_test_pred_with_actual.csv", index = False)