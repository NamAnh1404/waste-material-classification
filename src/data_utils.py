import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as res_preprocess

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, SEED


def get_preprocess_fn(model_type):
    if model_type == "cnn":
        return None
    if model_type == "effnet":
        return eff_preprocess
    if model_type == "resnet":
        return res_preprocess
    return None


def get_train_datagen(model_type, strong_aug=True):
    preprocess_fn = get_preprocess_fn(model_type)

    if model_type == "cnn":
        if strong_aug:
            return ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=25,
                width_shift_range=0.15,
                height_shift_range=0.15,
                zoom_range=0.25,
                horizontal_flip=True,
                brightness_range=(0.8, 1.2),
                fill_mode="nearest"
            )
        else:
            return ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=10,
                zoom_range=0.1,
                horizontal_flip=True
            )

    if strong_aug:
        return ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.25,
            shear_range=0.15,
            horizontal_flip=True,
            brightness_range=(0.8, 1.2),
            fill_mode="nearest"
        )
    else:
        return ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            rotation_range=10,
            zoom_range=0.1,
            horizontal_flip=True
        )


def get_test_datagen(model_type):
    preprocess_fn = get_preprocess_fn(model_type)

    if model_type == "cnn":
        return ImageDataGenerator(rescale=1.0 / 255)

    return ImageDataGenerator(preprocessing_function=preprocess_fn)


def make_generators(model_type, strong_aug=True):
    train_datagen = get_train_datagen(model_type, strong_aug=strong_aug)
    test_datagen = get_test_datagen(model_type)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=SEED
    )

    val_gen = test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen