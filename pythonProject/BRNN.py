def BLSTM():
    from keras.layes import Embedding, SimpleRNN
    model = Sequential()
    model.add(Embedding(input_dim, output_dim,, input_length=maxlen))
    model.add(Bidirectional(SimpleRNN(31, input_shape=(None,1,1,56), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, go_backwards=True, stateful=False, unroll=False)))
    model.add(Reshape((31 , )))
    from keras.layers import Dense
    from keras import backend as K

    model.add(Dense(31 , activation=K.softmax))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd')
    model.fit(a,b, epochs=5, batch_size=40)
    return model