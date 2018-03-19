from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Sequential
from keras.utils import np_utils
import pandas as pd
import wandb
import numpy as np
import cv2
from wandb.wandb_keras import WandbKerasCallback

run = wandb.init()
config = run.config
# parameters
config.batch_size = 32
config.num_epochs = 10
config.dense_layer_size = 100
config.img_width = 48
config.img_height = 48
config.first_layer_conv_width = 3
config.first_layer_conv_height = 3
input_shape = (48, 48, 1)

wandb_callback=  WandbKerasCallback(save_model=False)


def load_fer2013():
    
    data = pd.read_csv("fer2013/fer2013.csv")
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (width, height))
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()

    val_faces = faces[int(len(faces) * 0.8):]
    val_emotions = emotions[int(len(faces) * 0.8):]
    train_faces = faces[:int(len(faces) * 0.8)]
    train_emotions = emotions[:int(len(faces) * 0.8)]
    
    return train_faces, train_emotions, val_faces, val_emotions

# loading dataset
train_faces, train_emotions, val_faces, val_emotions = load_fer2013()


#reshape input data
#train_faces = train_faces.reshape(train_faces.shape[0], config.img_width, config.img_height, 1)
#val_faces = val_faces.reshape(val_faces.shape[0], config.img_width, config.img_height, 1)



train_faces = train_faces.astype('float32')
train_faces /= 255.
val_faces = val_faces.astype('float32')
val_faces /= 255.
# one hot encode outputs
#train_emotions = np_utils.to_categorical(train_emotions)
#val_emotions = np_utils.to_categorical(val_emotions)

num_samples = train_emotions.shape[0]
num_classes = train_emotions.shape[1]


#num_classes = val_emotions.shape[1]


model = Sequential()

model.add(Conv2D(32,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    input_shape=(48, 48, 1),
    activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.4))



model.add(Flatten())
model.add(Dense(config.dense_layer_size, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation="softmax"))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])

model.fit(train_faces, train_emotions, batch_size=config.batch_size,
                    epochs=config.num_epochs, verbose=1, callbacks=[wandb_callback],
                    validation_data=(val_faces, val_emotions))

model.save("emotion.h5")



