from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_train_datagen():
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='constant',
        cval=0
    )

def get_val_test_datagen():
    return ImageDataGenerator(rescale=1./255)