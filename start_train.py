import numpy as np
import pickle
import sys
import pandas as pd
import os
import help_train

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

def get_XY(varname, dat, time_steps, len_skip = -1, len_output = -1):
    X = []
    Y = [] 
    if len_skip == -1:
        len_skip = time_steps
    if len_output == -1:
        len_output = time_steps
    range_var = data_dict_used_range[varname][1] - data_dict_used_range[varname][0]
    dat = [(val_scale - data_dict_used_range[varname][0]) / range_var for val_scale in dat]
    for i in range(0, len(dat), len_skip):
        x_vals = dat[i:min(i + time_steps, len(dat))]
        y_vals = dat[i + time_steps:i + time_steps + len_output]
        if len(x_vals) == time_steps and len(y_vals) == len_output:
            X.append(np.array(x_vals))
            Y.append(np.array(y_vals))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

ws_range = [2, 3, 4, 5, 10, 20, 30]
sf1, sf2 = 5, 5

if not os.path.isfile("data_range.csv"):
    mini_dict = dict()
    maxi_dict = dict()
    data_range_dict = {"var": [], "min": [], "max": []}
    for nf2 in [1]:
        for nf1 in range(5):
            ride_name = "csv_results/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/"
            varnames = os.listdir("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/")
            for v in varnames:
                if "time" in v:
                    continue
                var = v.replace("actual_", "")
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
    for var in mini_dict:
        data_range_dict["var"].append(var)
        data_range_dict["min"].append(mini_dict[var])
        data_range_dict["max"].append(maxi_dict[var])
    df_data_range_dict = pd.DataFrame(data_range_dict)
    df_data_range_dict.to_csv("data_range.csv", index = False)

df_data_range_dict = pd.read_csv("data_range.csv", index_col = False)
data_dict_used_range = dict()
for ix in range(len(df_data_range_dict["var"])):
    data_dict_used_range[df_data_range_dict["var"][ix]] = (df_data_range_dict["min"][ix], df_data_range_dict["max"][ix])

for nf1 in range(sf1):
    for nf2 in range(sf2):
        for filename in os.listdir("actual_train/" + str(nf1 + 1) + "/" + str(nf2 + 1)):

            varname = filename.replace("actual_train_", "")

            if "abs" not in varname:
                continue
            if "time" in varname:
                continue

            file_object_train = load_object("actual_train/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_train_" + varname) 
            file_object_val = load_object("actual_val/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_val_" + varname)
            file_object_test = load_object("actual/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/actual_" + varname)
        
            for ws_use in ws_range:

                x_train_all = []
                y_train_all = []

                x_train_val_all = []
                y_train_val_all = []

                for k in file_object_train:

                    x_train_part, y_train_part = get_XY(varname, file_object_train[k], ws_use, 1, ws_use)
                    
                    for ix in range(len(x_train_part)):
                        x_train_all.append(x_train_part[ix]) 
                        y_train_all.append(y_train_part[ix])
                        x_train_val_all.append(x_train_part[ix]) 
                        y_train_val_all.append(y_train_part[ix])

                x_train_all = np.array(x_train_all)
                y_train_all = np.array(y_train_all)
                
                x_train_all_short = []
                y_train_all_short = []

                x_train_val_all_short = []
                y_train_val_all_short = []

                for k in file_object_train:

                    x_train_part, y_train_part = get_XY(varname, file_object_train[k], ws_use, ws_use, ws_use)
                    
                    for ix in range(len(x_train_part)):
                        x_train_all_short.append(x_train_part[ix]) 
                        y_train_all_short.append(y_train_part[ix])
                        x_train_val_all_short.append(x_train_part[ix]) 
                        y_train_val_all_short.append(y_train_part[ix])

                x_train_all_short = np.array(x_train_all_short)
                y_train_all_short = np.array(y_train_all_short)
                
                x_test_all = []
                y_test_all = []

                for k in file_object_test:

                    x_test_part, y_test_part = get_XY(varname, file_object_test[k], ws_use, 1, ws_use)
                    
                    for ix in range(len(x_test_part)):
                        x_test_all.append(x_test_part[ix]) 
                        y_test_all.append(y_test_part[ix])

                x_test_all = np.array(x_test_all)
                y_test_all = np.array(y_test_all)
                
                x_test_all_short = []
                y_test_all_short = []

                for k in file_object_test:

                    x_test_part, y_test_part = get_XY(varname, file_object_test[k], ws_use, ws_use, ws_use)
                    
                    for ix in range(len(x_test_part)):
                        x_test_all_short.append(x_test_part[ix]) 
                        y_test_all_short.append(y_test_part[ix])

                x_test_all_short = np.array(x_test_all_short)
                y_test_all_short = np.array(y_test_all_short)
                
                x_val_all = []
                y_val_all = []

                for k in file_object_val:

                    x_val_part, y_val_part = get_XY(varname, file_object_val[k], ws_use, 1, ws_use)
                    
                    for ix in range(len(x_val_part)):
                        x_val_all.append(x_val_part[ix]) 
                        y_val_all.append(y_val_part[ix])
                        x_train_val_all.append(x_val_part[ix]) 
                        y_train_val_all.append(y_val_part[ix])

                x_val_all = np.array(x_val_all)
                y_val_all = np.array(y_val_all)

                x_train_val_all = np.array(x_train_val_all)
                y_train_val_all = np.array(y_train_val_all)
                
                x_val_all_short = []
                y_val_all_short = []

                for k in file_object_val:

                    x_val_part, y_val_part = get_XY(varname, file_object_val[k], ws_use, ws_use, ws_use)
                    
                    for ix in range(len(x_val_part)):
                        x_val_all_short.append(x_val_part[ix]) 
                        y_val_all_short.append(y_val_part[ix])
                        x_train_val_all_short.append(x_val_part[ix]) 
                        y_train_val_all_short.append(y_val_part[ix])

                x_val_all_short = np.array(x_val_all_short)
                y_val_all_short = np.array(y_val_all_short)

                x_train_val_all_short = np.array(x_train_val_all_short)
                y_train_val_all_short = np.array(y_train_val_all_short)

                for model_name in ["Bi", "Conv"]:

                    saving_dir = model_name + "/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + varname + "/" + str(ws_use)
                    saving_name = saving_dir + "/" + saving_dir.replace("/", "_")

                    if not os.path.isdir(saving_dir):
                        os.makedirs(saving_dir)

                    # Write output to file
                    sys.stdout = open(saving_name + ".txt", "w", encoding="utf-8")
                    
                    numcells = 32
                    kernelsize = 4
                    if model_name == "Bi":
                        help_train.new_train("val_loss", ws_use, numcells, -1, saving_name, x_train_val_all, y_train_val_all, x_val_all, y_val_all)
                    else:
                        help_train.new_train("val_loss", ws_use, numcells, kernelsize, saving_name, x_train_val_all, y_train_val_all, x_val_all, y_val_all)

                    # Close output file
                    sys.stdout.close()
                    
                    # Write output to file
                    sys.stdout = open(saving_name + "_test_output.txt", "w", encoding="utf-8")

                    help_train.new_test(varname, saving_name + ".h5", saving_name + "_test_pred.csv", x_test_all_short, y_test_all_short)
                    
                    # Close output file
                    sys.stdout.close()