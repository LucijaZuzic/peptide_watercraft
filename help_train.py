import tensorflow as tf
import pandas as pd
import model_create

LEARNING_RATE_SET = 0.01
MAX_DROPOUT = 0.5
MAX_BATCH_SIZE = 600
MAX_EPOCHS = 70

# This function keeps the initial learning rate for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def return_callbacks(model_file, metric):
    callbacks = [
        # Save the best model (the one with the lowest validation loss).
        tf.keras.callbacks.ModelCheckpoint(
            model_file, save_best_only=True, monitor=metric, mode="min"
        ),
        tf.keras.callbacks.LearningRateScheduler(scheduler),
    ]
    return callbacks

def new_test(best_model_file, pred_file, test_data, test_labels):
    # Load the best model.
    if best_model_file != "":
        best_model = tf.keras.models.load_model(best_model_file)

    # Get model predictions on the test data.
    model_predictions = best_model.predict(test_data, batch_size=MAX_BATCH_SIZE)
    # Get model predictions on the test data.
    dict_new_model = {"predicted": [], "actual": []}
    for ix1 in range(len(test_labels)):
        for ix2 in range(len(test_labels[ix1])):
            dict_new_model["predicted"].append(model_predictions[ix1][ix2])
            dict_new_model["actual"].append(test_labels[ix1][ix2])
    df_new_model = pd.DataFrame(dict_new_model)
    df_new_model.to_csv(pred_file, index = False)

def new_train(metric, num_cells_param, conv_kernel_size_param, model_name, train_and_validation_data, train_and_validation_labels, val_data, val_labels):
    
    if conv_kernel_size_param > 0:
        model = model_create.seq_model(
            conv1_filters=64,
            conv2_filters=64,
            conv_kernel_size=conv_kernel_size_param,
            num_cells=num_cells_param // 2,
            dropout=MAX_DROPOUT
        )
    else:
        model = model_create.one_prop_model(
            lstm1=5, 
            lstm2=5, 
            dense=num_cells_param // 2, 
            dropout=MAX_DROPOUT, 
            lambda2=0.0
        )
    # Save graphical representation of the model to a file.
    #tf.keras.utils.plot_model(model, to_file=model_name + ".png", show_shapes=True)

    # Print model summary.
    model.summary()

    callbacks = return_callbacks(model_name + ".h5", metric)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_SET)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    
    history = model.fit(
        train_and_validation_data,
        train_and_validation_labels,
        validation_data=[val_data, val_labels],
        epochs=MAX_EPOCHS,
        batch_size=MAX_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    dict_hist = dict()
    for history_part in ["val_loss", "val_acc", "loss", "acc"]:
        if history_part not in history.history:
            continue
        dict_hist[history_part] = list(history.history[history_part])
    print(dict_hist)
    df_hist = pd.DataFrame(dict_hist)
    df_hist.to_csv(model_name + ".csv", index = False)