import os
import pandas as pd
import pickle
import numpy as np

def load_object(file_name): 
    with open(file_name, 'rb') as file_object:
        data = pickle.load(file_object) 
        file_object.close()
        return data
    
def get_XY(dat, time_steps, len_skip = -1, len_output = -1):
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
            X.append(np.array(x_vals))
            Y.append(np.array(y_vals))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y
    
sf1, sf2 = 5, 5

for nf1 in [2]:

    for nf2 in [0]:

        ride_name = "csv_results/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"

        varnames = os.listdir("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/")
        actual_var = dict()
        for v in varnames:
            var = v.replace("actual_", "")
            actual_var[var] = load_object("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + v)
            
        for varname in actual_var:
            if "time" in varname:
                continue
            for model_name in ["Bi", "Conv"]:
                for ws in [2, 3, 4, 5, 10, 20, 30]:
                        saving_dir = model_name + "/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + str(ws)
                        saving_name = saving_dir + "/" + saving_dir.replace("/", "_")
                        pd_file = pd.read_csv(saving_name + "_test_pred.csv")
                        all_actual_actual = []
                        so_far = 0

                        for ride in actual_var[varname]:

                            x_test_all_short = []
                            y_test_all_short = []

                            x_test_part, y_test_part = get_XY(actual_var[varname][ride], ws, ws, ws)
                            
                            for ix in range(len(x_test_part)):
                                x_test_all_short.extend(x_test_part[ix]) 
                                y_test_all_short.extend(y_test_part[ix])
                            
                            dict_write = dict()

                            dict_write = {"predicted": list(pd_file["predicted"][so_far:so_far+len(y_test_all_short)]),
                                          "predicted_descaled": list(pd_file["predicted_descaled"][so_far:so_far+len(y_test_all_short)]),
                                          "actual": list(pd_file["actual"][so_far:so_far+len(y_test_all_short)]),
                                          "actual_descaled": list(pd_file["actual_descaled"][so_far:so_far+len(y_test_all_short)]),
                                          "actual_actual": y_test_all_short}
                            
                            so_far += len(y_test_all_short)

                            all_actual_actual.extend(y_test_all_short)

                            ride_name = "csv_results/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + ride.replace(".csv", "/").replace("/cleaned_csv/", "/") + str(varname) + "/" + model_name + "/" 
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