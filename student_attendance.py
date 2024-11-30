import numpy as np
import cv2
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 64
DATASET_PATH = 'student_attendance/Dataset'  

def load_data(dataset_path, img_size):
    X = []
    y = []
    label_dict = {}
    label = 0

    for student in os.listdir(dataset_path):
        student_path = os.path.join(dataset_path, student)
        if os.path.isdir(student_path):
            label_dict[label] = student  # Store mapping of label to student name
            for img_file in os.listdir(student_path):
                img_path = os.path.join(student_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))  
                X.append(img)
                y.append(label)
            label += 1

    X = np.array(X).astype('float32') / 255.0  
    y = np.array(y)
    return X, y, label_dict

X, y, label_dict = load_data(DATASET_PATH, IMG_SIZE)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

# reshape for CNN input (since it's grayscale images)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# convert labels to categorical (one-hot encoding)
num_classes = len(label_dict)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train.argmax(axis=1)),
    y=y_train.argmax(axis=1)
)

class_weight_dict = dict(enumerate(class_weights))

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,   
    width_shift_range=0.1, 
    height_shift_range=0.1,
    zoom_range=0.2,     
    horizontal_flip=True
)
datagen.fit(X_train)

def build_face_recognition_model(input_shape, num_classes):
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Second Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),

        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

input_shape = (IMG_SIZE, IMG_SIZE, 1)  
face_recognition_model = build_face_recognition_model(input_shape, num_classes)

face_recognition_model.summary()

epochs = 30 
batch_size = 32  
history = face_recognition_model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs, 
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict  
)

test_loss, test_accuracy = face_recognition_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

model_json = face_recognition_model.to_json()
with open("face_recognition_model.json", "w") as json_file:
    json_file.write(model_json)

face_recognition_model.save_weights("face_recognition_weights.weights.h5")

