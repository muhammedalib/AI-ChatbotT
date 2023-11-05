from tensorflow.python.keras.layers import LSTM, Dropout, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adam

from Training import classes, dataset_generator, documents
import tensorflow as tf

def create_lstm_model():
    model = Sequential()
    model.add(LSTM(
        units=64, input_shape=(None, 300), return_sequences=True))
    model.add(Dropout(0.2))
    sgd = SGD(learning_rate=1e-3 , decay=1e-6,momentum=0.9,nesterov=True)
    model.add(Dense(units=len(classes), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                   optimizer=sgd(learning_rate = 1e-3),
                   metrics=['accuracy'])
    model.summary()
    return model


model = create_lstm_model()


def train_model(dataset):
    train_ds = tf.data.Dataset.from_generator(dataset_generator(dataset),
                                              (tf.float32, tf.float32),
                                              output_shapes=(
                                                  (None, None, 300), (None, None, len(classes))),
                                              args=[False])
    model.fit(train_ds, epochs=25, shuffle=False)


train_model(documents)


model.save('chatbot_model.model')