from keras.layers import Input, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Flatten, Concatenate
from keras.models import Model
from keras import regularizers


def get_model_1():
    inputs = Input(shape=(6912,))
    model = Dense(3951, activation='relu')(inputs)
    model = Dropout(0.5)(model)
    model = Dense(1024, activation='relu')(model)
    model = Dropout(0.3)(model)
    model = Dense(512, activation='relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(64, activation='relu')(model)
    model = Dropout(0.2)(model)
    predictions = Dense(4, activation='softmax')(model)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_2():
    """This is regularised"""
    inputs = Input(shape=(6912,))
    model = Dense(3951, activation='relu')(inputs)
    model = Dropout(0.5)(model)
    model = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(model)
    model = Dropout(0.3)(model)
    model = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(model)
    model = Dropout(0.3)(model)
    model = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(model)
    model = Dropout(0.3)(model)
    predictions = Dense(4, activation='softmax')(model)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


NUMBER_OF_FILTERS = [300, 200]
KERNEL_SIZE = [4, 5]
HIDDEN_LAYER = 300
SIMILARITY_LAYER = 20


def model_cnn():
    # 1D Conv Layer with multiple possible kernel sizes
    inputs = Input(shape=(3, 2304))

    model = Conv1D(filters=300,
                   kernel_size=3,
                   padding='valid',
                   activation='relu',
                   kernel_regularizer=regularizers.l2(0.001),
                   strides=1)(inputs)
    model = GlobalMaxPooling1D()(model)

    flat_input = Flatten()(inputs)
    flat_input = Dense(1024, activation='relu',
                       kernel_regularizer=regularizers.l2(0.01),
                       activity_regularizer=regularizers.l2(0.01))(flat_input)
    flat_input = Dropout(0.5)(flat_input)

    flat_input = Dense(512, activation='relu',
                       kernel_regularizer=regularizers.l2(0.01),
                       activity_regularizer=regularizers.l2(0.01))(flat_input)
    flat_input = Dropout(0.5)(flat_input)

    model = Concatenate()([model, flat_input])

    model = Dense(264, activation='relu', kernel_regularizer=regularizers.l2(0.01))(model)
    model = Dropout(0.5)(model)
    model = Dense(64, activation='relu',
                  kernel_regularizer=regularizers.l2(0.01),
                  activity_regularizer=regularizers.l2(0.01))(model)
    model = Dropout(0.3)(model)
    predictions = Dense(4, activation='softmax')(model)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
