from keras.layers import Input, Dense, Dropout
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
    model = Dense(1024, activation='relu')(model)
    model = Dropout(0.3)(model)
    model = Dense(512, activation='relu', activity_regularizer=regularizers.l2(0.001))(model)
    model = Dropout(0.2)(model)
    model = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.01))(model)
    model = Dropout(0.2)(model)
    predictions = Dense(4, activation='softmax')(model)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
