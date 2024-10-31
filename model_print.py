
import os
import tensorflow as tf

def new_test(best_model_file):
    # Load the best model.
    print(ftotal)
    if best_model_file != "":
        best_model = tf.keras.models.load_model(best_model_file)
    for i, layer in enumerate (best_model.layers):
        print (i, layer)
        try:
            print ("    ",layer.activation)
        except AttributeError:
            print('   no activation attribute')
    
for model in ["Bi", "Conv"]:
    for n1 in os.listdir(model):
        for n2 in os.listdir(model + "/" + str(n1)):
            for var in os.listdir(model + "/" + str(n1) + "/" + str(n2)):
                for ws in os.listdir(model + "/" + str(n1) + "/" + str(n2) + "/" + var):
                    fdir = model + "/" + str(n1) + "/" + str(n2) + "/" + var + "/" + ws
                    fnm = fdir.replace("/", "_")
                    ftotal = fdir + "/" + fnm + ".h5"
                    new_test(ftotal)
                    break
                break
            break
        break