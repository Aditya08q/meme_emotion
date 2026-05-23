
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks, optimizers
import tensorflow as tf
import json


TRAIN_DIR = "data/train"
TEST_DIR  = "data/test"
MODEL_DIR = "models"
IMG_SIZE = (48,48)
BATCH_SIZE = 64
EPOCHS = 30
NUM_CLASSES = 5  

os.makedirs(MODEL_DIR, exist_ok=True)

def build_model(input_shape=(48,48,1), num_classes=7):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
    
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, color_mode='grayscale',
        batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
    )
    val_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, color_mode='grayscale',
        batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )

   
    print("Class indices:", train_gen.class_indices)

    
    with open(os.path.join(MODEL_DIR, 'label_map.json'), 'w') as f:
     json.dump(train_gen.class_indices, f)
     print("Saved label mapping to model/label_map.json")

    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), num_classes=train_gen.num_classes)
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint = callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, 'emotion_model.h5'),
                                           save_best_only=True, monitor='val_accuracy', mode='max')
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=EPOCHS,
                        callbacks=[checkpoint, reduce_lr, early])
    

    
    model.save(os.path.join(MODEL_DIR, 'emotion_model_final.h5'))
    print("Training complete. Models saved in", MODEL_DIR)

if __name__ == "__main__":
    main()
