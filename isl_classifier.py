#%%
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

#%%
class Classifier:
    def __init__(self, train_path, val_path):
        #can make horizontal_flip=False if required
        train_datagen = ImageDataGenerator(rescale=1./255,
                                           shear_range=0.1,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           horizontal_flip=True)
        self.train = train_datagen.flow_from_directory(train_path,
                                                       target_size=(64, 64),
                                                       batch_size=24,
                                                       color_mode='grayscale',
                                                       class_mode='categorical',
                                                       shuffle=True)
        
        test_datagen = ImageDataGenerator(rescale=1./255,
                                          shear_range=0.1,
                                          width_shift_range=0.1,
                                          height_shift_range=0.1,
                                          horizontal_flip=True)
        self.test = test_datagen.flow_from_directory(val_path,
                                                     target_size=(64, 64),
                                                     batch_size=24,
                                                     color_mode='grayscale',
                                                     class_mode='categorical',
                                                     shuffle=True)

    def architecture(self):
        self.cnn = tf.keras.models.Sequential()
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 1)))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
        self.cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
        self.cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
        self.cnn.add(tf.keras.layers.Flatten())
        self.cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))
        self.cnn.add(tf.keras.layers.Dropout(0.2))
        self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.cnn.add(tf.keras.layers.Dropout(0.3))
        self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.cnn.add(tf.keras.layers.Dense(units=36, activation='softmax'))
       
    def summary(self):
        self.cnn.summary()
        
    def train_model(self, epochs=20):
        #checkpoint file saved with epoch_no, val_accuracy, learning_rate
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')        
        checkpoint_path = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}-{lr:.6f}.hdf5"
        checkpoints = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        self.cnn.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        self.cnn.fit(x=self.train, validation_data=self.test, epochs=epochs, callbacks=[reduce_lr, early_stop, checkpoints])
    
    def load_model(self, path):
        path_param = path.split('-')
        self.cnn = load_model(path)
        self.current_epoch = int(path_param[2])
        self.current_learning = float(path_param[4][0:-5])
        
    def continue_training(self, epochs=20):
        #checkpoint file saved with epoch_no, val_accuracy, learning_rate
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
        checkpoint_path = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}-{lr:.6f}.hdf5"
        checkpoints = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        self.cnn.compile(optimizer=Adam(learning_rate=self.current_learning), loss='categorical_crossentropy', metrics=['accuracy'])
        self.cnn.fit(x=self.train, validation_data=self.test, epochs=epochs, initial_epoch=self.current_epoch, callbacks=[reduce_lr, early_stop, checkpoints])
    
    def sample_prediction(self):
        word_dict = {0:'1', 1:'3', 2:'4', 3:'5', 4:'6', 5:'7', 6:'8', 7:'9', 8:'a', 9:'b', 10:'break', 11:'c', 12:'d',
                     13:'e', 14:'f', 15:'g', 16:'h', 17:'i', 18:'j', 19:'k', 20:'l', 21:'m', 22:'n', 23:'o', 24:'p',
                     25:'q', 26:'r', 27:'s', 28:'space', 29:'t', 30:'u', 31:'v', 32:'w', 33:'x', 34:'y', 35:'z'}
        imgs, labels = next(self.test)
        predictions = self.cnn.predict(imgs, verbose=0)
        print('Predictions on a small set of test data.')
        print('')
        print('Predicted Labels')
        for ind, i in enumerate(predictions):
            print(word_dict[np.argmax(i)], end='   ')
        print('\nActual labels')
        for i in labels:
            print(word_dict[np.argmax(i)], end='   ')
        scores = self.cnn.evaluate(imgs, labels, verbose=0)
        print('\n'+f'{self.cnn.metrics_names[0]} of {scores[0]}; {self.cnn.metrics_names[1]} of {scores[1]*100}%')
                
#%%

tr=r'C:\Users\Dell\3D Objects\Indian_sign_language\Dataset\train'
tes=r'C:\Users\Dell\3D Objects\Indian_sign_language\Dataset\test'

ISL_v1 = Classifier(tr,tes)
ISL_v1.architecture()
ISL_v1.summary()
ISL_v1.train_model()
ISL_v1.sample_prediction()

#%%
load_path='weights-improvement-06-0.82-0.000200.hdf5'
ISL_v1 = Classifier(r'C:\Users\Dell\3D Objects\Indian_sign_language\Dataset\train', r'C:\Users\Dell\3D Objects\Indian_sign_language\Dataset\test')
ISL_v1.load_model(load_path)
ISL_v1.continue_training()
ISL_v1.sample_prediction()

