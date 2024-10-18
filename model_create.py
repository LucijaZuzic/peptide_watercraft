from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D
from tensorflow.keras.regularizers import l2

def seq_model(
    conv1_filters=64,
    conv2_filters=64,
    conv_kernel_size=6,
    num_cells=64,
    dropout=0.1
):
    
    model_input = Input(shape=(None, 1), name="input_1")

    if conv1_filters > 0:
        x = Conv1D(
            conv1_filters,
            conv_kernel_size,
            padding="same",
            kernel_initializer="he_normal",
            name="conv1d_1",
        )(model_input)

        if conv2_filters > 0:
            x = Conv1D(
                conv2_filters,
                conv_kernel_size,
                padding="same",
                kernel_initializer="he_normal",
                name="conv1d_2",
            )(x)
        x = Bidirectional(LSTM(num_cells, name="bi_lstm"))(x)
    else:
        x = Bidirectional(LSTM(num_cells, name="bi_lstm"))(model_input)

    if dropout > 0:
        x = Dropout(dropout, name="dropout")(x)

    return Model(inputs=model_input, outputs=x)

def one_prop_model(lstm1=5, lstm2=5, dense=15, dropout=0.2, lambda2=0.0):

    # Instantiate separate submodels.
    inputs = []
    outputs = []
    
    input1 = Input((None, 1))
    lstm_layer_1 = Bidirectional(LSTM(lstm1, return_sequences=True))(input1)
    lstm_layer_2 = LSTM(lstm2)(lstm_layer_1)

    dense_layer1 = Dense(dense, activation="selu", kernel_regularizer=l2(l=lambda2))(lstm_layer_2)
    output1 = Dropout(dropout)(dense_layer1)
    inputs.append(input1)
    outputs.append(output1)
 
    return Model(inputs=inputs, outputs=outputs)