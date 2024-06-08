
def CNN_LSTM():
    model = Sequential()
    model.add(Conv2D(168, 4, strides=(1, 168), padding='same', input_shape=(1,1,336)))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=None, padding='valid' , dim_ordering="th"))
    model.add(Reshape((1,56)))
    model.add(LSTM(31, input_shape=(None,1,1,56), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, go_backwards=True, stateful=False, unroll=False))
    model.add(Reshape((31 , )))
    from keras.layers import Dense
    from keras import backend as K

    model.add(Dense(31 , activation=K.softmax))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd')
    model.fit(a,b, epochs=5, batch_size=40)
    return model