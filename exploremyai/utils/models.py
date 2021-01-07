from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping


def create_lstm_regressor(lstm_cells, input_timesteps):
    """This function returns a Keras model object of a single-layer acceptor LSTM. It is meant for the single-step
    prediction of a univariate series with input window size input_timesteps
    """
    model = Sequential()
    model.add(LSTM(lstm_cells, input_shape=(input_timesteps, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='RMSprop')
    return model


def create_lstm_classifier(lstm_cells, input_timesteps, class_count):
    """This function returns a Keras model object of a single-layer acceptor LSTM. It is meant for the single-step
    prediction of a one-hot encoded univariate series with input window size input_timesteps and whose one-hot encoded
    label values are in the range [0, class_count)
    """
    model = Sequential()
    model.add(LSTM(lstm_cells, input_shape=(input_timesteps, class_count)))
    model.add(Dense(class_count), activation='softmax')
    model.compile(loss='sparse_categorical_crossentropy', optimizer='RMSprop')
    return model


def train_model(model, X, y, batch_size=1, epochs=1000, verbose=0):
    """This function trains (fits) and returns a keras model with early stopping enabled. It also returns the number of
    epochs trained before stopping
    """
    callback = [EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)]
    history = model.fit(X, y, batch_size, epochs, verbose, callbacks=callback)
    # print('History length', len(history.history['loss']))
    return model, len(history.history['loss'])