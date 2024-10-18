import numpy as np
import pickle
import sys
import os
import help_train

def load_object(file_name): 
    with open(file_name, 'rb') as file_object:
        data = pickle.load(file_object) 
        file_object.close()
        return data
    
def save_object(file_name, std1):       
    with open(file_name, 'wb') as file_object:
        pickle.dump(std1, file_object) 
        file_object.close()

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

ws_range = [2, 3, 4, 5, 10, 20, 30]
ws_range = [2, 3, 4, 10]
ws_range = [2, 4, 10, 20, 30]
ws_range = [2, 4, 10]
ws_range = [10]
sf1, sf2 = 5, 5
sf1, sf2 = 1, 1
import keras; print(keras.__version__)
for nf1 in range(sf1):
    for nf2 in range(sf2):
        for filename in os.listdir("actual_train/" + str(nf1 + 1) + "/" + str(nf2 + 1)):

            varname = filename.replace("actual_train_", "")

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

                    x_train_part, y_train_part = get_XY(file_object_train[k], ws_use, 1, ws_use)
                    
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

                    x_train_part, y_train_part = get_XY(file_object_train[k], ws_use, ws_use, ws_use)
                    
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

                    x_test_part, y_test_part = get_XY(file_object_test[k], ws_use, 1, ws_use)
                    
                    for ix in range(len(x_test_part)):
                        x_test_all.append(x_test_part[ix]) 
                        y_test_all.append(y_test_part[ix])

                x_test_all = np.array(x_test_all)
                y_test_all = np.array(y_test_all)
                
                x_test_all_short = []
                y_test_all_short = []

                for k in file_object_test:

                    x_test_part, y_test_part = get_XY(file_object_test[k], ws_use, ws_use, ws_use)
                    
                    for ix in range(len(x_test_part)):
                        x_test_all_short.append(x_test_part[ix]) 
                        y_test_all_short.append(y_test_part[ix])

                x_test_all_short = np.array(x_test_all_short)
                y_test_all_short = np.array(y_test_all_short)
                
                x_val_all = []
                y_val_all = []

                for k in file_object_val:

                    x_val_part, y_val_part = get_XY(file_object_val[k], ws_use, 1, ws_use)
                    
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

                    x_val_part, y_val_part = get_XY(file_object_val[k], ws_use, ws_use, ws_use)
                    
                    for ix in range(len(x_val_part)):
                        x_val_all_short.append(x_val_part[ix]) 
                        y_val_all_short.append(y_val_part[ix])
                        x_train_val_all_short.append(x_val_part[ix]) 
                        y_train_val_all_short.append(y_val_part[ix])

                x_val_all_short = np.array(x_val_all_short)
                y_val_all_short = np.array(y_val_all_short)

                x_train_val_all_short = np.array(x_train_val_all_short)
                y_train_val_all_short = np.array(y_train_val_all_short)

                for model_name in ["Conv"]:
                    import tensorflow as tf
                    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
                    print(np.shape(x_train_val_all), np.shape(y_train_val_all))
                    print(np.shape(x_val_all), np.shape(y_val_all))
                    print(np.shape(x_test_all), np.shape(y_test_all))

                    saving_dir = model_name + "/" + varname + "/" + str(ws_use)
                    saving_name = saving_dir + "/" + saving_dir.replace("/", "_")

                    if not os.path.isdir(saving_dir):
                        os.makedirs(saving_dir)

                    # Write output to file
                    sys.stdout = open(saving_name + ".txt", "w", encoding="utf-8")
 
                    if model_name == "Bi":
                        help_train.new_train("val_loss", ws_use, -1, saving_name, x_train_val_all, y_train_val_all, x_val_all, y_val_all)
                    else:
                        help_train.new_train("val_loss", ws_use, 4, saving_name, x_train_val_all, y_train_val_all, x_val_all, y_val_all)

                    # Close output file
                    sys.stdout.close()

                    help_train.new_test(saving_name + ".h5", saving_name + "_test_pred.csv", x_test_all_short, y_test_all_short)