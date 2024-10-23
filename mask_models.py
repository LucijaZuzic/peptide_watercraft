from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D, Masking
from tensorflow.keras.regularizers import l2

def _create_seq_model(
    final_size,
    input_shape,
    conv1_filters=64,
    conv2_filters=64,
    conv_kernel_size=4,
    num_cells=15,
    dropout=0.5,
    mask_value=2
):
    model_input = Input((None, 1))
    mask = Masking(mask_value=mask_value)(model_input)

    if conv1_filters > 0:
        x = Conv1D(
            conv1_filters,
            conv_kernel_size,
            padding="same",
            kernel_initializer="he_normal"
        )(mask)

        if conv2_filters > 0:
            x = Conv1D(
                conv2_filters,
                conv_kernel_size,
                padding="same",
                kernel_initializer="he_normal"
            )(x)
        x = Bidirectional(LSTM(num_cells))(x)
    else:
        x = Bidirectional(LSTM(num_cells))(mask)

    if dropout > 0:
        x = Dropout(dropout)(x)

    x = Dense(final_size, activation="sigmoid")(x)

    return model_input, x

def multi_seq_model(
    num_props,
    final_size,
    input_shape,
    conv1_filters=64,
    conv2_filters=64,
    conv_kernel_size=4,
    num_cells=15,
    dropout=0.5,
    mask_value=2
):
    # Instantiate separate submodels.
    inputs = []
    outputs = []
    for i in range(num_props):
        input1, output1 = _create_seq_model(final_size, input_shape, conv1_filters, conv2_filters, conv_kernel_size, num_cells, dropout, mask_value)
        inputs.append(input1)
        outputs.append(output1)

    return Model(inputs=inputs, outputs=outputs)

def _one_prop_model(final_size, lstm1=5, lstm2=5, dense=15, dropout=0.5, lambda2=0.0, mask_value=2):
    # LSTM model which processes dipeptide AP scores.

    input_layer = Input((None, 1))
    mask = Masking(mask_value=mask_value)(input_layer)
    lstm_layer_1 = Bidirectional(LSTM(lstm1, return_sequences=True))(mask)
    lstm_layer_2 = LSTM(lstm2)(lstm_layer_1)

    dense_layer1 = Dense(dense, activation="selu", kernel_regularizer=l2(l=lambda2))(
        lstm_layer_2
    )
    if dropout > 0:
        dropout_1 = Dropout(dropout)(dense_layer1)
        final_output_layer = Dense(final_size, activation="sigmoid")(dropout_1)
    else:
        final_output_layer = Dense(final_size, activation="sigmoid")(dense_layer1)
    return input_layer, final_output_layer

def only_amino_di_tri_model(num_props, final_size, lstm1=5, lstm2=5, dense=15, dropout=0.5, lambda2=0.0, mask_value=2):
    # Instantiate separate submodels.
    inputs = []
    outputs = []
    for i in range(num_props):
        input1, output1 = _one_prop_model(final_size, lstm1, lstm2, dense, dropout, lambda2, mask_value)
        inputs.append(input1)
        outputs.append(output1)

    return Model(inputs=inputs, outputs=outputs)
