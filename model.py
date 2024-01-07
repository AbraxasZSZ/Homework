from keras import Input
from keras.src.layers import Dropout, Dense
from keras.models import *
from gap_load import X_train, y_train, X_test
import numpy as np

np.random.seed(2024)

input_tensor = Input(X_train.shape[1:])
x = Dropout(0.5)(input_tensor)
x = Dense(1, activation='sigmoid')(x)
model = Model(input_tensor, x)

model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, nb_epoch=8, validation_split=0.2)
model.save('model.h5')

y_pred = model.predict(X_test, verbose=1)
y_pred = y_pred.clip(min=0.005, max=0.995)
