import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A
from tensorflow.keras import datasets, layers, models
import numpy as np
!pip install git+https://github.com/mjkvaak/ImageDataAugmentor
from ImageDataAugmentor.image_data_augmentor import *
import seaborn as sns
import pandas as pd

"""# Code"""

def getData(tranch: int, test_size = 0.33) -> pd.DataFrame:
  from sklearn.model_selection import train_test_split

  labels_path = '/content/drive/My Drive/M499/tranch'+str(tranch)+'_labels.csv'
  labels = pd.read_csv(labels_path)

  if tranch == 1:
    from zipfile import ZipFile
    pictures_path = '/content/drive/My Drive/M499/persons-posture-tranch'+str(tranch)+'.zip'
    zip_file = ZipFile(pictures_path)
    file_list = [obj.filename for obj in zip_file.infolist()]
    file_list_simple = [name.split('/')[-1] for name in file_list]
    
    names = pd.DataFrame({'file_path': file_list, 'file_name': file_list_simple}) 
    df = pd.merge(names, labels, on = 'file_name')
    df.columns = df.columns.str.replace("file_path","final_url")
    df.drop("file_name", inplace=True, axis='columns')
    zip_file.close()
  else:
    df = labels

  df = df.dropna()
  df.index = range(len(df))

  train, test = train_test_split(df, test_size=0.33, random_state=42)

  train = train.query("primary_posture != 'Unknown'")
  test = test.query("primary_posture != 'Unknown'")

  return train, test

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
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

def getGenerator( train: pd.DataFrame, test:  pd.DataFrame, 
                 tranch_image_path: str,
                 model_preprocess: callable,
                 eraser: callable = get_random_eraser,
                 AUGMENTATIONS: "augment_func" = AUGMENTATIONS
                 ):
  
  
  custom_preprocess = lambda img: model_preprocess(eraser()(img))
  
  

  train_datagen = ImageDataAugmentor(data_format="channels_last",
                              augment = AUGMENTATIONS,
                              preprocess_input = custom_preprocess,
                              validation_split=0.2,
  )

  val_datagen = ImageDataAugmentor(data_format="channels_last",
                              validation_split=0.2,
                              preprocess_input= model_preprocess
  )

  test_datagen = ImageDataAugmentor(data_format="channels_last",
                              preprocess_input= model_preprocess
  )

  train_gen = train_datagen.flow_from_dataframe(train,
                              directory = tranch_image_path,
                              x_col = "final_url",
                              y_col = "primary_posture",
                              class_mode="sparse",
                              batch_size = 32,
                              seed=42,
                              subset = "training",
                              shuffle=True,
                              target_size=(224, 224),
                              validate_filenames = True,
  )

  val_gen = val_datagen.flow_from_dataframe(train,
                              directory = tranch_image_path,
                              x_col = "final_url",
                              y_col = "primary_posture",
                              class_mode="sparse",
                              batch_size = 32,
                              seed=42,
                              subset = "validation",
                              shuffle=True,
                              target_size=(224, 224),
                              validate_filenames = True,
  )


  test_gen = test_datagen.flow_from_dataframe(test,
                              directory = tranch_image_path,
                              x_col = "final_url",
                              y_col = "primary_posture",
                              class_mode="sparse",
                              batch_size = 32,
                              seed=42,
                              shuffle=True,
                              target_size=(224, 224),
                              validate_filenames = True,
  )

  return train_gen, val_gen, test_gen

class trainModel():
  def __init__(self, modelstr: str):
    
    assert modelstr in ["MobileNetV2", "EfficientNetB0"], "only support mobilenetv2 and efficientnetB0"
    self.modelstr = modelstr
    self.model = None

  def newModel(self)-> tf.keras.Model:
    from tensorflow.keras.layers import (GlobalAveragePooling2D, 
      Dropout, Dense, BatchNormalization)
    from tensorflow.keras import optimizers, losses

    if self.modelstr == "MobileNetV2":
      from tensorflow.keras.applications import MobileNetV2
      base_model=MobileNetV2(weights='imagenet',include_top=False,
                            input_shape = (224, 224, 3))
      base_model.trainable = True
      inputs = tf.keras.Input(shape=(224, 224, 3), name = "image_input")
      x = base_model(inputs, training=True, )
      x=GlobalAveragePooling2D()(x)
      x = BatchNormalization()(x)
      x = Dropout(0.3)(x)
      x=Dense(1024,activation='relu')(x)
      x=Dense(1024,activation='relu')(x)
      x=Dense(512,activation='relu')(x)
      x=Dense(128,activation='relu')(x)
      preds=Dense(3,activation='softmax')(x)
 
      self.model= tf.keras.Model(inputs=inputs,outputs=[preds], name = "mobileNetV2")
      self.model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=False),
          optimizer=optimizers.Adam(learning_rate=1e-3),
          metrics=['accuracy'])

    else: # EfficientNet
      from tensorflow.keras.applications import EfficientNetB0

      def unfreeze_model(model):
          # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
          for layer in model.layers[-20:]:
              if not isinstance(layer, layers.BatchNormalization):
                  layer.trainable = True

      inputs = tf.keras.Input(shape=(224, 224, 3), name = "image_input")
      conv_base = EfficientNetB0(input_shape=(224, 224,3), 
                                 input_tensor= inputs, drop_connect_rate=0.4,
                            include_top = False)

      conv_base.trainable = False

      unfreeze_model(conv_base)
      x= GlobalAveragePooling2D()(conv_base.output)
      x = BatchNormalization()(x)
      x = Dropout(0.3)(x)
      x=Dense(512,activation='relu')(x)
      x=Dense(256,activation='relu')(x)
      x=Dense(128,activation='relu')(x)
      preds=Dense(3,activation='softmax')(x)
      
      self.model= tf.keras.Model(inputs=inputs,outputs=[preds], name = "efficientNetB0")
      self.model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=False),
          optimizer=optimizers.Adam(learning_rate=1e-3),
          metrics=['accuracy'])

    return self.model
  
  def getPreprocessFunc(self) -> callable:
    from tensorflow.keras.applications import mobilenet_v2
    from tensorflow.keras.applications import efficientnet
    return mobilenet_v2.preprocess_input if self.modelstr == "MobileNetV2" else efficientnet.preprocess_input

  def plot(self):
    assert self.model is not None, "run newModel first"
    return tf.keras.utils.plot_model(self.model)

  def summary(self):
    assert self.model is not None, "run newModel first"
    return self.model.summary()

def train_model(model, train_gen, val_gen, test_gen,
                save_dir: str, epochs = 60):
  from datetime import datetime
  import json
  
  # Reduce learning rate when there is a change lesser than <min_delta> in <val_accuracy> for more than <patience> epochs
  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy',
                                                  mode = 'max',
                                                  min_delta = 0.01,
                                                  patience = 3,
                                                  factor = 0.25,
                                                  verbose = 1,
                                                  cooldown = 0,
                                                  min_lr = 0.00000001)
 
  # Stop the training process when there is a change lesser than <min_delta> in <val_accuracy> for more than <patience> epochs
  early_stopper = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy',
                                                  mode = 'max',
                                                  min_delta = 0.005,
                                                  patience = 10,
                                                  verbose = 1,
                                                  restore_best_weights = True)
  
  txt_log = open(save_dir+".log", mode='wt', buffering=1)

  log_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end = lambda epoch, logs: txt_log.write(
      json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
    on_train_end = lambda logs: txt_log.close()
  )

  his = model.fit(train_gen, 
      steps_per_epoch=len(train_gen),
      epochs=epochs,
      callbacks=[
                 tf.keras.callbacks.ModelCheckpoint(filepath=save_dir + '.{epoch:02d}-{val_accuracy:.2f}.h5'),
                 early_stopper, reduce_lr, log_callback],
      verbose = 1,
      # class_weight = class_weights,
      validation_data = val_gen,
      validation_steps = len(val_gen),
      shuffle = True,
  )
  plt.plot(his.history["accuracy"], label = "training")
  plt.plot(his.history["val_accuracy"], label = "validating")
  plt.legend()
  plt.savefig(save_dir + "plot.jpg")
  
  cur = datetime.now()
  save_dir += str(cur.day) + "-" + str(cur.hour) + ".h5"
 
  model.save(save_dir)

def train(ensemble_num = 15):
  import os
  base_dir = "/content/drive/My Drive/"
  for modelstr in ["EfficientNetB0", "MobileNetV2"]:
    print("++++++++++++++++++++++++++++++++++++++++++++++")
    print("running for model", modelstr)
    print("++++++++++++++++++++++++++++++++++++++++++++++")
    MODEL = trainModel(modelstr)
    for tranch in range(1, 4):
        print(f"~~~~~~      tranch {tranch}       ~~~~~~~")
        print()
        img_dir = os.path.join(base_dir, f"M499/tranch{tranch}")
        gens = getGenerator(
                            *getData(tranch), # tranch train & test
                            tranch_image_path = img_dir,
                            model_preprocess = MODEL.getPreprocessFunc(),
                            eraser = get_random_eraser,
                            AUGMENTATIONS = AUGMENTATIONS
                        )
        for n in range(ensemble_num):
            for gen in gens:
              gen.reset()
            
            save_dir = os.path.join(base_dir, f"model/{tranch}/{modelstr}-{n}-")
            
            train_model(MODEL.newModel(), 
                        *gens,
                        save_dir = save_dir,
                        epochs = 1   
                        )
          
train(1)