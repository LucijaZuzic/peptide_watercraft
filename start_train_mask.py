import numpy as np
import pickle
import sys
import pandas as pd
import os
import help_train

MASK_VAL = 2
MAXINT = 10 ** 6

def load_object(file_name): 
    with open(file_name, 'rb') as file_object:
        data = pickle.load(file_object) 
        file_object.close()
        return data
    
def save_object(file_name, std1):       
    with open(file_name, 'wb') as file_object:
        pickle.dump(std1, file_object) 
        file_object.close()

def get_XY(varname, dat, time_steps, len_skip = -1, len_output = -1, max_len = 30, mask_value = 2):
    X = []
    Y = [] 
    if len_skip == -1:
        len_skip = time_steps
    if len_output == -1:
        len_output = time_steps
    range_var = data_dict_used_range_abs[varname][1] - data_dict_used_range_abs[varname][0]
    dat = [(val_scale - data_dict_used_range_abs[varname][0]) / range_var for val_scale in dat]
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

ws_range = [2, 3, 4, 5, 10, 20, 30]
sf1, sf2 = 5, 5

if not os.path.isfile("data_range_abs.csv"):
    mini_dict = dict()
    maxi_dict = dict()
    data_range_abs_dict = {"var": [], "min": [], "max": []}
    for nf2 in [1]:
        for nf1 in range(5):
            ride_name = "csv_results/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"
            varnames = os.listdir("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/")
            for v in varnames:
                if "time" in v:
                    continue
                var = v.replace("actual_", "")
                if "abs" not in var:
                    if var not in mini_dict:
                        mini_dict[var] = MAXINT
                    if var not in maxi_dict:
                        maxi_dict[var] = -MAXINT
                    actual_var_par = load_object("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + v)
                    for r in actual_var_par:
                        vals = actual_var_par[r]
                        mini_tmp = min(vals)
                        maxi_tmp = max(vals)
                        if mini_tmp < mini_dict[var]:
                            mini_dict[var] = mini_tmp
                        if maxi_tmp > maxi_dict[var]:
                            maxi_dict[var] = maxi_tmp
                else:
                    if var + "_abs" not in mini_dict:
                        mini_dict[var + "_abs"] = MAXINT
                    if var + "_abs" not in maxi_dict:
                        maxi_dict[var + "_abs"] = -MAXINT
                    if var + "_sgn" not in mini_dict:
                        mini_dict[var + "_sgn"] = MAXINT
                    if var + "_sgn" not in maxi_dict:
                        maxi_dict[var + "_sgn"] = -MAXINT
                    actual_var_par = load_object("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + v)
                    for r in actual_var_par:
                        vals = actual_var_par[r]
                        vals_abs = [abs(vv) for vv in vals]
                        mini_tmp_abs = min(vals_abs)
                        maxi_tmp_abs = max(vals_abs)
                        if mini_tmp_abs < mini_dict[var + "_abs"]:
                            mini_dict[var + "_abs"] = mini_tmp_abs
                        if maxi_tmp_abs > maxi_dict[var + "_abs"]:
                            maxi_dict[var + "_abs"] = maxi_tmp_abs
                        vals_sgn = [int(vv > 0) for vv in vals]
                        mini_tmp_sgn = min(vals_sgn)
                        maxi_tmp_sgn = max(vals_sgn)
                        if mini_tmp_sgn < mini_dict[var + "_sgn"]:
                            mini_dict[var + "_sgn"] = mini_tmp_sgn
                        if maxi_tmp_sgn > maxi_dict[var + "_sgn"]:
                            maxi_dict[var + "_sgn"] = maxi_tmp_sgn

    for var in mini_dict:
        data_range_abs_dict["var"].append(var)
        data_range_abs_dict["min"].append(mini_dict[var])
        data_range_abs_dict["max"].append(maxi_dict[var])
    df_data_range_abs_dict = pd.DataFrame(data_range_abs_dict)
    df_data_range_abs_dict.to_csv("data_range_abs.csv", index = False)

df_data_range_abs_dict = pd.read_csv("data_range_abs.csv", index_col = False)
data_dict_used_range_abs = dict()
for ix in range(len(df_data_range_abs_dict["var"])):
    data_dict_used_range_abs[df_data_range_abs_dict["var"][ix]] = (df_data_range_abs_dict["min"][ix], df_data_range_abs_dict["max"][ix])

for nf1 in range(sf1):
    for nf2 in range(sf2):
            
        file_object_train = dict()
        file_object_val = dict()
        file_object_test = dict()

        for filename in os.listdir("actual_train/" + str(nf1 + 1) + "/" + str(nf2 + 1)):

            varname = filename.replace("actual_train_", "")

            if "time" in varname:
                continue
            if "abs" not in varname:
                file_object_train[varname] = load_object("actual_train/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_train_" + varname) 
                file_object_val[varname] = load_object("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_val_" + varname)
                file_object_test[varname] = load_object("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_" + varname)
            else:
                file_object_train_tmp = load_object("actual_train/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_train_" + varname) 
                file_object_val_tmp = load_object("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_val_" + varname)
                file_object_test_tmp = load_object("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_" + varname)
                file_object_train_abs = dict() 
                file_object_val_abs = dict() 
                file_object_test_abs = dict() 
                file_object_train_sgn = dict() 
                file_object_val_sgn = dict() 
                file_object_test_sgn = dict() 
                for k in file_object_train_tmp:
                    file_object_train_abs[k] = [abs(vv) for vv in file_object_train_tmp[k]]
                    file_object_train_sgn[k] = [int(vv > 0) for vv in file_object_train_tmp[k]]
                for k in file_object_val_tmp:
                    file_object_val_abs[k] = [abs(vv) for vv in file_object_val_tmp[k]]
                    file_object_val_sgn[k] = [int(vv > 0) for vv in file_object_val_tmp[k]]
                for k in file_object_test_tmp:
                    file_object_test_abs[k] = [abs(vv) for vv in file_object_test_tmp[k]]
                    file_object_test_sgn[k] = [int(vv > 0) for vv in file_object_test_tmp[k]]
                file_object_train[varname + "_abs"] = file_object_train_abs
                file_object_val[varname + "_abs"] = file_object_val_abs
                file_object_test[varname + "_abs"] = file_object_test_abs
                file_object_train[varname + "_sgn"] = file_object_train_sgn
                file_object_val[varname + "_sgn"] = file_object_val_sgn
                file_object_test[varname + "_sgn"] = file_object_test_sgn

        ord_var = sorted(list(file_object_train.keys()))

        for ws_use in ws_range:

            x_train_all = []
            y_train_all = []
            x_train_all_var = {v: [] for v in ord_var}
            y_train_all_var = {v: [] for v in ord_var}

            x_train_val_all = []
            y_train_val_all = []
            x_train_val_all_var = {v: [] for v in ord_var}
            y_train_val_all_var = {v: [] for v in ord_var}

            for k in file_object_train["speed"]:

                mini_len = 1000000
                for varname in ord_var:
                    if len(file_object_train[varname][k]) < mini_len:
                        mini_len = len(file_object_train[varname][k])
                x_train_part, y_train_part = dict(), dict()
                for varname in ord_var:
                    x_train_part[varname], y_train_part[varname] = get_XY(varname, file_object_train[varname][k][:mini_len], ws_use, 1, ws_use, ws_use, MASK_VAL)
                for ix in range(len(x_train_part["speed"])):
                    x_train_all.append([])
                    y_train_all.append([])
                    x_train_val_all.append([])
                    y_train_val_all.append([])
                    for varname in ord_var:
                        x_train_all[-1].append(x_train_part[varname][ix]) 
                        y_train_all[-1].append(y_train_part[varname][ix])
                        x_train_val_all[-1].append(x_train_part[varname][ix]) 
                        y_train_val_all[-1].append(y_train_part[varname][ix])
                        x_train_all_var[varname].append(x_train_part[varname][ix]) 
                        y_train_all_var[varname].append(y_train_part[varname][ix])
                        x_train_val_all_var[varname].append(x_train_part[varname][ix]) 
                        y_train_val_all_var[varname].append(y_train_part[varname][ix])
                    x_train_all[-1] = np.array(x_train_all[-1])
                    y_train_all[-1] = np.array(y_train_all[-1])
                    x_train_val_all[-1] = np.array(x_train_val_all[-1])
                    y_train_val_all[-1] = np.array(y_train_val_all[-1])

            x_train_all = np.array(x_train_all)
            y_train_all = np.array(y_train_all)
            for varname in ord_var:
                x_train_all_var[varname] = np.array(x_train_all_var[varname])
                y_train_all_var[varname] = np.array(y_train_all_var[varname])
            
            x_train_all_short = []
            y_train_all_short = []
            x_train_all_short_var = {v: [] for v in ord_var}
            y_train_all_short_var = {v: [] for v in ord_var}

            x_train_val_all_short = []
            y_train_val_all_short = []
            x_train_val_all_short_var = {v: [] for v in ord_var}
            y_train_val_all_short_var = {v: [] for v in ord_var}

            for k in file_object_train["speed"]:

                mini_len = 1000000
                for varname in ord_var:
                    if len(file_object_train[varname][k]) < mini_len:
                        mini_len = len(file_object_train[varname][k])
                x_train_part, y_train_part = dict(), dict()
                for varname in ord_var:
                    x_train_part[varname], y_train_part[varname] = get_XY(varname, file_object_train[varname][k][:mini_len], ws_use, ws_use, ws_use, ws_use, MASK_VAL)
                for ix in range(len(x_train_part["speed"])):
                    x_train_all_short.append([])
                    y_train_all_short.append([])
                    x_train_val_all_short.append([])
                    y_train_val_all_short.append([])
                    for varname in ord_var:
                        x_train_all_short[-1].append(x_train_part[varname][ix]) 
                        y_train_all_short[-1].append(y_train_part[varname][ix])
                        x_train_val_all_short[-1].append(x_train_part[varname][ix]) 
                        y_train_val_all_short[-1].append(y_train_part[varname][ix])
                        x_train_all_short_var[varname].append(x_train_part[varname][ix]) 
                        y_train_all_short_var[varname].append(y_train_part[varname][ix])
                        x_train_val_all_short_var[varname].append(x_train_part[varname][ix]) 
                        y_train_val_all_short_var[varname].append(y_train_part[varname][ix])
                    x_train_all_short[-1] = np.array(x_train_all_short[-1])
                    y_train_all_short[-1] = np.array(y_train_all_short[-1])
                    x_train_val_all_short[-1] = np.array(x_train_val_all_short[-1])
                    y_train_val_all_short[-1] = np.array(y_train_val_all_short[-1])

            x_train_all_short = np.array(x_train_all_short)
            y_train_all_short = np.array(y_train_all_short)
            for varname in ord_var:
                x_train_all_short_var[varname] = np.array(x_train_all_short_var[varname])
                y_train_all_short_var[varname] = np.array(y_train_all_short_var[varname])
            
            x_test_all = []
            y_test_all = []
            x_test_all_var = {v: [] for v in ord_var}
            y_test_all_var = {v: [] for v in ord_var}

            for k in file_object_test["speed"]:

                mini_len = 1000000
                for varname in ord_var:
                    if len(file_object_test[varname][k]) < mini_len:
                        mini_len = len(file_object_test[varname][k])
                x_test_part, y_test_part = dict(), dict()
                for varname in ord_var:
                    x_test_part[varname], y_test_part[varname] = get_XY(varname, file_object_test[varname][k][:mini_len], ws_use, 1, ws_use, ws_use, MASK_VAL)
                for ix in range(len(x_test_part["speed"])):
                    x_test_all.append([])
                    y_test_all.append([])
                    for varname in ord_var:
                        x_test_all[-1].append(x_test_part[varname][ix]) 
                        y_test_all[-1].append(y_test_part[varname][ix])
                        x_test_all_var[varname].append(x_test_part[varname][ix]) 
                        y_test_all_var[varname].append(y_test_part[varname][ix])
                    x_test_all[-1] = np.array(x_test_all[-1])
                    y_test_all[-1] = np.array(y_test_all[-1])

            x_test_all = np.array(x_test_all)
            y_test_all = np.array(y_test_all)
            for varname in ord_var:
                x_test_all_var[varname] = np.array(x_test_all_var[varname])
                y_test_all_var[varname] = np.array(y_test_all_var[varname])
            
            x_test_all_short = []
            y_test_all_short = []
            x_test_all_short_var = {v: [] for v in ord_var}
            y_test_all_short_var = {v: [] for v in ord_var}

            for k in file_object_test["speed"]:

                mini_len = 1000000
                for varname in ord_var:
                    if len(file_object_test[varname][k]) < mini_len:
                        mini_len = len(file_object_test[varname][k])
                x_test_part, y_test_part = dict(), dict()
                for varname in ord_var:
                    x_test_part[varname], y_test_part[varname] = get_XY(varname, file_object_test[varname][k][:mini_len], ws_use, ws_use, ws_use, ws_use, MASK_VAL)
                for ix in range(len(x_test_part["speed"])):
                    x_test_all_short.append([])
                    y_test_all_short.append([])
                    for varname in ord_var:
                        x_test_all_short[-1].append(x_test_part[varname][ix]) 
                        y_test_all_short[-1].append(y_test_part[varname][ix])
                        x_test_all_short_var[varname].append(x_test_part[varname][ix]) 
                        y_test_all_short_var[varname].append(y_test_part[varname][ix])
                    x_test_all_short[-1] = np.array(x_test_all_short[-1])
                    y_test_all_short[-1] = np.array(y_test_all_short[-1])

            x_test_all_short = np.array(x_test_all_short)
            y_test_all_short = np.array(y_test_all_short)
            for varname in ord_var:
                x_test_all_short_var[varname] = np.array(x_test_all_short_var[varname])
                y_test_all_short_var[varname] = np.array(y_test_all_short_var[varname])
             
            x_val_all = []
            y_val_all = []
            x_val_all_var = {v: [] for v in ord_var}
            y_val_all_var = {v: [] for v in ord_var}

            for k in file_object_val["speed"]:

                mini_len = 1000000
                for varname in ord_var:
                    if len(file_object_val[varname][k]) < mini_len:
                        mini_len = len(file_object_val[varname][k])
                x_val_part, y_val_part = dict(), dict()
                for varname in ord_var:
                    x_val_part[varname], y_val_part[varname] = get_XY(varname, file_object_val[varname][k][:mini_len], ws_use, 1, ws_use, ws_use, MASK_VAL)
                for ix in range(len(x_val_part["speed"])):
                    x_val_all.append([])
                    y_val_all.append([])
                    x_train_val_all.append([])
                    y_train_val_all.append([])
                    for varname in ord_var:
                        x_val_all[-1].append(x_val_part[varname][ix]) 
                        y_val_all[-1].append(y_val_part[varname][ix])
                        x_train_val_all[-1].append(x_val_part[varname][ix]) 
                        y_train_val_all[-1].append(y_val_part[varname][ix])
                        x_val_all_var[varname].append(x_val_part[varname][ix]) 
                        y_val_all_var[varname].append(y_val_part[varname][ix])
                        x_train_val_all_var[varname].append(x_val_part[varname][ix]) 
                        y_train_val_all_var[varname].append(y_val_part[varname][ix])
                    x_val_all[-1] = np.array(x_val_all[-1])
                    y_val_all[-1] = np.array(y_val_all[-1])
                    x_train_val_all[-1] = np.array(x_train_val_all[-1])
                    y_train_val_all[-1] = np.array(y_train_val_all[-1])

            x_val_all = np.array(x_val_all)
            y_val_all = np.array(y_val_all)
            for varname in ord_var:
                x_val_all_var[varname] = np.array(x_val_all_var[varname])
                y_val_all_var[varname] = np.array(y_val_all_var[varname])
            
            x_train_val_all = np.array(x_train_val_all)
            y_train_val_all = np.array(y_train_val_all)
            for varname in ord_var:
                x_train_val_all_var[varname] = np.array(x_train_val_all_var[varname])
                y_train_val_all_var[varname] = np.array(y_train_val_all_var[varname])

            x_val_all_short = []
            y_val_all_short = []
            x_val_all_short_var = {v: [] for v in ord_var}
            y_val_all_short_var = {v: [] for v in ord_var}

            for k in file_object_val["speed"]:

                mini_len = 1000000
                for varname in ord_var:
                    if len(file_object_val[varname][k]) < mini_len:
                        mini_len = len(file_object_val[varname][k])
                x_val_part, y_val_part = dict(), dict()
                for varname in ord_var:
                    x_val_part[varname], y_val_part[varname] = get_XY(varname, file_object_val[varname][k][:mini_len], ws_use, ws_use, ws_use, ws_use, MASK_VAL)
                for ix in range(len(x_val_part["speed"])):
                    x_val_all_short.append([])
                    y_val_all_short.append([])
                    x_train_val_all_short.append([])
                    y_train_val_all_short.append([])
                    for varname in ord_var:
                        x_val_all_short[-1].append(x_val_part[varname][ix]) 
                        y_val_all_short[-1].append(y_val_part[varname][ix])
                        x_train_val_all_short[-1].append(x_val_part[varname][ix]) 
                        y_train_val_all_short[-1].append(y_val_part[varname][ix])
                        x_val_all_short_var[varname].append(x_val_part[varname][ix]) 
                        y_val_all_short_var[varname].append(y_val_part[varname][ix])
                        x_train_val_all_short_var[varname].append(x_val_part[varname][ix]) 
                        y_train_val_all_short_var[varname].append(y_val_part[varname][ix])
                    x_val_all_short[-1] = np.array(x_val_all_short[-1])
                    y_val_all_short[-1] = np.array(y_val_all_short[-1])
                    x_train_val_all_short[-1] = np.array(x_train_val_all_short[-1])
                    y_train_val_all_short[-1] = np.array(y_train_val_all_short[-1])

            x_val_all_short = np.array(x_val_all_short)
            y_val_all_short = np.array(y_val_all_short)
            for varname in ord_var:
                x_val_all_short_var[varname] = np.array(x_val_all_short_var[varname])
                y_val_all_short_var[varname] = np.array(y_val_all_short_var[varname])

            x_train_val_all_short = np.array(x_train_val_all_short)
            y_train_val_all_short = np.array(y_train_val_all_short)
            for varname in ord_var:
                x_train_val_all_short_var[varname] = np.array(x_train_val_all_short_var[varname])
                y_train_val_all_short_var[varname] = np.array(y_train_val_all_short_var[varname])
                
            x_train_val_all_short_merged = [np.array(x_train_val_all_short_var[varname])  for varname in ord_var]
            y_train_val_all_short_merged = [np.array(y_train_val_all_short_var[varname])  for varname in ord_var]
            x_train_val_all_merged = [np.array(x_train_val_all_var[varname])  for varname in ord_var]
            y_train_val_all_merged = [np.array(y_train_val_all_var[varname])  for varname in ord_var]
            
            x_train_all_short_merged = [np.array(x_train_all_short_var[varname])  for varname in ord_var]
            y_train_all_short_merged = [np.array(y_train_all_short_var[varname])  for varname in ord_var]
            x_train_all_merged = [np.array(x_train_all_var[varname])  for varname in ord_var]
            y_train_all_merged = [np.array(y_train_all_var[varname])  for varname in ord_var]

            x_test_all_short_merged = [np.array(x_test_all_short_var[varname])  for varname in ord_var]
            y_test_all_short_merged = [np.array(y_test_all_short_var[varname])  for varname in ord_var]
            x_test_all_merged = [np.array(x_test_all_var[varname])  for varname in ord_var]
            y_test_all_merged = [np.array(y_test_all_var[varname])  for varname in ord_var]
            
            x_val_all_short_merged = [np.array(x_val_all_short_var[varname])  for varname in ord_var]
            y_val_all_short_merged = [np.array(y_val_all_short_var[varname])  for varname in ord_var]
            x_val_all_merged = [np.array(x_val_all_var[varname])  for varname in ord_var]
            y_val_all_merged = [np.array(y_val_all_var[varname])  for varname in ord_var]

            for model_name in ["Bi_Mask", "Conv_Mask"]:

                saving_dir = model_name + "/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + str(ws_use)
                saving_name = saving_dir + "/" + saving_dir.replace("/", "_")

                if not os.path.isdir(saving_dir):
                    os.makedirs(saving_dir)

                # Write output to file
                sys.stdout = open(saving_name + ".txt", "w", encoding="utf-8")

                numcells = 35
                kernelsize = 4

                if model_name == "Bi_Mask":
                    help_train.mask_train("val_loss", MASK_VAL, ws_use, numcells, -1, saving_name, x_train_val_all_merged, y_train_val_all_merged, x_val_all_merged, y_val_all_merged)
                else:
                    help_train.mask_train("val_loss", MASK_VAL, ws_use, numcells, kernelsize, saving_name, x_train_val_all_merged, y_train_val_all_merged, x_val_all_merged, y_val_all_merged)

                # Close output file
                sys.stdout.close()

                # Write output to file
                sys.stdout = open(saving_name + "_test_output.txt", "w", encoding="utf-8")

                help_train.new_test_mask(ord_var, MASK_VAL, saving_name + ".h5", saving_name + "_test_pred.csv", x_test_all_short_merged, y_test_all_short_merged)
                
                # Close output file
                sys.stdout.close()