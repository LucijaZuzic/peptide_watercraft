import os
import pandas as pd
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

sf1, sf2 = 5, 5

var_list = os.listdir("Bi/1/1/")
model_list = ["Bi", "Conv"]
ws_list = [2, 3, 4, 5, 10, 20, 30]

data_frame_val_new  = {"variable": [], "model": [], "ws": [], "test": [], "val": [], "R2": [], "MAE": [], "MSE": [], "RMSE": []}

pd_file = pd.read_csv("data_frame_val_old.csv", index_col = False)

data_frame_val_merged = dict()
data_frame_val_merged["variable"] = list(pd_file["variable"])
data_frame_val_merged["model"] = list(pd_file["model"])
data_frame_val_merged["ws"] = list(pd_file["ws"])
data_frame_val_merged["test"] = list(pd_file["test"])
data_frame_val_merged["val"] = list(pd_file["val"])
data_frame_val_merged["R2"] = list(pd_file["R2"])
data_frame_val_merged["MAE"] = list(pd_file["MAE"])
data_frame_val_merged["MSE"] = list(pd_file["MSE"])
data_frame_val_merged["RMSE"] = list(pd_file["RMSE"])

for var in var_list:
    for model in model_list:
        for ws in ws_list:
            for nf1 in range(sf1):
                for nf2 in range(sf2):
                    print(var, model, ws, nf1, nf2)
                    saving_dir_mask = model + "_Mask/" + str(nf1 + 1) + "/" + str(nf2 + 1) + "/" + str(ws)
                    saving_name_mask  = saving_dir_mask+ "/" + saving_dir_mask.replace("/", "_")
                    if "abs" not in var:
                        if os.path.isfile(saving_name_mask + "_" + var + "_test_pred.csv"):
                            pd_file_mask = pd.read_csv(saving_name_mask + "_" + var + "_test_pred.csv", index_col = False)
                            predicted = pd_file_mask["predicted_descaled"]
                            actual = pd_file_mask["actual_descaled"]
                            r2_pred = r2_score(actual, predicted)
                            mae_pred = mean_absolute_error(actual, predicted)
                            mse_pred = mean_squared_error(actual, predicted)
                            rmse_pred = math.sqrt(mse_pred)
                            print(r2_pred, mae_pred, mse_pred, rmse_pred)
                    else:
                        if os.path.isfile(saving_name_mask + "_" + var + "_abs_test_pred.csv"):
                            pd_file_mask_abs = pd.read_csv(saving_name_mask + "_" + var + "_abs_test_pred.csv", index_col = False)
                            pd_file_mask_sgn = pd.read_csv(saving_name_mask + "_" + var + "_sgn_test_pred.csv", index_col = False)
                            predicted_abs = pd_file_mask_abs["predicted_descaled"]
                            actual_abs = pd_file_mask_abs["actual_descaled"]
                            predicted_sgn = pd_file_mask_sgn["predicted_descaled"]
                            actual_sgn = pd_file_mask_sgn["actual_descaled"]
                            predicted = [(int(predicted_sgn[ix_val] > 0.5) * 2 - 1) * predicted_abs[ix_val] for ix_val in range(len(predicted_abs))]
                            actual = [(int(actual_sgn[ix_val] > 0.5) * 2 - 1) * actual_abs[ix_val] for ix_val in range(len(actual_abs))]
                            r2_pred = r2_score(actual, predicted)
                            mae_pred = mean_absolute_error(actual, predicted)
                            mse_pred = mean_squared_error(actual, predicted)
                            rmse_pred = math.sqrt(mse_pred)
                            print(r2_pred, mae_pred, mse_pred, rmse_pred)
                        data_frame_val_new["variable"].append(var)
                        data_frame_val_new["model"].append(model)
                        data_frame_val_new["ws"].append(ws)
                        data_frame_val_new["test"].append(nf1 + 1)
                        data_frame_val_new["val"].append(nf2 + 1)
                        data_frame_val_new["R2"].append(r2_pred)
                        data_frame_val_new["MAE"].append(mae_pred)
                        data_frame_val_new["MSE"].append(mse_pred)
                        data_frame_val_new["RMSE"].append(rmse_pred)
                        data_frame_val_merged["variable"].append(var)
                        data_frame_val_merged["model"].append(model)
                        data_frame_val_merged["ws"].append(ws)
                        data_frame_val_merged["test"].append(nf1 + 1)
                        data_frame_val_merged["val"].append(nf2 + 1)
                        data_frame_val_merged["R2"].append(r2_pred)
                        data_frame_val_merged["MAE"].append(mae_pred)
                        data_frame_val_merged["MSE"].append(mse_pred)
                        data_frame_val_merged["RMSE"].append(rmse_pred)

df_data_frame_val_new = pd.DataFrame(data_frame_val_new)
df_data_frame_val_new.to_csv("data_frame_val_mask_new.csv", index = False)

df_data_frame_val_merged = pd.DataFrame(data_frame_val_merged)
df_data_frame_val_merged.to_csv("data_frame_val_mask_merged.csv", index = False)