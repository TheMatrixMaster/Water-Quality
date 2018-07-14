
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn import preprocessing, model_selection

X = np.load('D:/Hackerfest2018/Processed_Data/training_data.npy')
new_X = []
for image in X:
	new_X.append(image/255)

X = np.array(new_X)

y = np.load('D:/Hackerfest2018/Processed_Data/training_labels.npy')
# X_val = np.load('D:/Hackerfest2018/Processed_Data/validation_data.npy') 

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#X_test = np.load('D:/Hackerfest2018/Processed_Data/testing_data.npy')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-hackathon2018-2.h5', verbose=1, save_best_only=True)

model.fit(X_train, y_train, batch_size=256, epochs=10, validation_split=0.2, callbacks=[earlystopper, checkpointer])

scores = model.evaluate(X_test, y_test)
print(f"Accuracy: {scores[1]} %")
