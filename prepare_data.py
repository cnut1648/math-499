from sklearn.model_selection import train_test_split
from config import label_dir, zip_dir, BATCH_SIZE
import os
import pandas as pd
import numpy as np
import albumentations as A

from ImageDataAugmentor.image_data_augmentor import *


def getData(tranch: int, test_size=0.33) -> pd.DataFrame:
    labels = pd.read_csv(os.path.join(label_dir, f"tranch{tranch}_labels.csv"))
    df = labels
    if tranch == 1:
        df.columns = df.columns.str.replace("file_name", "final_url")

    df = df.dropna()
    df.index = range(len(df))

    train, test = train_test_split(df, test_size=test_size, random_state=42)

    train = train.query("primary_posture != 'Unknown'")
    test = test.query("primary_posture != 'Unknown'")

    return train, test


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape

        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w] = c

        return input_img

    return eraser


AUGMENTATIONS = A.Compose([
    A.Rotate(limit=40),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)
    ]),
    A.OneOf([
        A.ElasticTransform(alpha=224, sigma=224 * 0.05, alpha_affine=224 * 0.03),
        A.GridDistortion(),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
    ], p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.RandomContrast(limit=0.2, p=0.5),
    A.HorizontalFlip(),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10),
])


def getGenerator(train: pd.DataFrame, test: pd.DataFrame,
                 tranch_image_path: str,
                 model_preprocess: callable,
                 eraser: callable = get_random_eraser,
                 AUGMENTATIONS: "augment_func" = AUGMENTATIONS
                 ):
    custom_preprocess = lambda img: model_preprocess(eraser()(img))

    train_datagen = ImageDataAugmentor(data_format="channels_last",
                                       augment=AUGMENTATIONS,
                                       preprocess_input=custom_preprocess,
                                       validation_split=0.2,
                                       )

    val_datagen = ImageDataAugmentor(data_format="channels_last",
                                     validation_split=0.2,
                                     preprocess_input=model_preprocess
                                     )

    test_datagen = ImageDataAugmentor(data_format="channels_last",
                                      preprocess_input=model_preprocess
                                      )

    train_gen = train_datagen.flow_from_dataframe(train,
                                                  directory=tranch_image_path,
                                                  x_col="final_url",
                                                  y_col="primary_posture",
                                                  class_mode="sparse",
                                                  batch_size=BATCH_SIZE,
                                                  seed=42,
                                                  subset="training",
                                                  shuffle=True,
                                                  target_size=(224, 224),
                                                  validate_filenames=True,
                                                  )

    val_gen = val_datagen.flow_from_dataframe(train,
                                              directory=tranch_image_path,
                                              x_col="final_url",
                                              y_col="primary_posture",
                                              class_mode="sparse",
                                              batch_size=BATCH_SIZE,
                                              seed=42,
                                              subset="validation",
                                              shuffle=True,
                                              target_size=(224, 224),
                                              validate_filenames=True,
                                              )

    test_gen = test_datagen.flow_from_dataframe(test,
                                                directory=tranch_image_path,
                                                x_col="final_url",
                                                y_col="primary_posture",
                                                class_mode="sparse",
                                                batch_size=BATCH_SIZE,
                                                seed=42,
                                                shuffle=True,
                                                target_size=(224, 224),
                                                validate_filenames=True,
                                                )

    return train_gen, val_gen, test_gen
