import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, GlobalMaxPooling1D, Softmax, Lambda, Multiply, Concatenate
from tensorflow.keras.models import Model

# Define the input size
# Define the input size
input_shape = (336, 40)
output_shape = (31, 40)

# Create the CNN model
inputs = tf.keras.Input(shape=input_shape)
x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
cnn_output = Dense(output_shape[0] * output_shape[1], activation='relu')(x)
cnn_output = tf.keras.layers.Reshape(output_shape)(cnn_output)
# Create the location-based attention mechanism
attention_inp = tf.keras.Input(shape=output_shape)
location_attention = Dense(output_shape[0])(attention_inp)
location_attention = Softmax()(location_attention)
location_attention = Lambda(lambda x: tf.expand_dims(x, axis=2))(location_attention)
location_attention = Lambda(lambda x: tf.tile(x, [1, 1, output_shape[1]]))(location_attention)

# Apply attention to the CNN output
attention_mul = Multiply()([cnn_output, location_attention])
attention_mul = Concatenate(axis=2)([attention_mul, attention_inp])
attention_mul = Dense(output_shape[1], activation='relu')(attention_mul)

# Create the final model
model = Model(inputs=[inputs, attention_inp], outputs=attention_mul)

# Compile and print the model summary
model.compile(optimizer='adam', loss='mse')
model.summary()

#  Step 4: Train the model
# model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))
#
# # Step 5: Evaluate the model
# loss, accuracy = model.evaluate(x_test, y_test)

# Predict on the test set
#y_pred = model.predict(X_test)