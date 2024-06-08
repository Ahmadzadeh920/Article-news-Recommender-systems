

from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Attention, Bidirectional, LSTM, Dense , Resizing
from tensorflow.keras.models import Model
from initial_run import x_train , x_tst , y_train , y_tst
input_shape = (336,40)  # Input size
output_shape = (40, 1,31)  # Output size
input_layer = Input(shape=input_shape)
conv1d_layer = Conv1D(62, kernel_size=3, activation='relu', padding='same')(input_layer)
pooling_layer = MaxPooling1D(pool_size=8)(conv1d_layer)
attention_layer = Attention()([pooling_layer, pooling_layer])
bi_lstm_layer = Bidirectional(LSTM(16, return_sequences=True))(attention_layer)
output_layer = Dense(units=output_shape[2], activation='softmax')(bi_lstm_layer)
resize_layer = Resizing(height=40 , width=1 )(output_layer)
model = Model(inputs=input_layer, outputs=resize_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# configure the learning process of a neural network model
model.compile(loss='categorical_crossentropy',
                  optimizer='sgd')
# Step 4: Train the model
import numpy as np
import tensorflow as tf
for i in range(0, len(x_train)):
    y_resize= [[[0 for i in range(31)] for j in range(output_shape[1])] for k in range(40)]
    x= np.array(x_train[i])
    y = np.array(y_train[i]).transpose()
    for j in range(0, len(y)):
        y_resize[j][0] = y[j]
    y_resize = np.array(y_resize)
    model.fit(x , y_resize, epochs= 100)



# Step 5: Test the model

#predictions = model.predict(X_test)