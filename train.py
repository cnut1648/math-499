import tensorflow as tf
import numpy as np
from ImageDataAugmentor.image_data_augmentor import *
import pandas as pd
import config
from prepare_data import getData, get_random_eraser, AUGMENTATIONS, getGenerator
from tensorflow.keras.layers import (GlobalAveragePooling2D, 
            Dropout, Dense, BatchNormalization)
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers, losses
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.applications import EfficientNetB0, efficientnet
from datetime import datetime
import json
import os


class trainModel():
  def __init__(self, modelstr: str):
    
    assert modelstr in ["MobileNetV2", "EfficientNetB0"], "only support mobilenetv2 and efficientnetB0"
    self.modelstr = modelstr
    self.model = None

  def newModel(self)-> tf.keras.Model:
    if self.modelstr == "MobileNetV2":
      
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
    return mobilenet_v2.preprocess_input if self.modelstr == "MobileNetV2" else efficientnet.preprocess_input

  def plot(self):
    assert self.model is not None, "run newModel first"
    return tf.keras.utils.plot_model(self.model)

  def summary(self):
    assert self.model is not None, "run newModel first"
    return self.model.summary()



def train_model(model, train_gen, val_gen, test_gen,
                save_dir: str, epochs = 60):
  
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
  plt.savefig(save_dir + ".jpg")
  
  cur = datetime.now()
  save_dir += f":::final--{cur.day}:{cur.hour}.h5"
  model.save(save_dir)

def train(ensemble_num = 15):
  for modelstr in ["EfficientNetB0", "MobileNetV2"]:
    print("++++++++++++++++++++++++++++++++++++++++++++++")
    print("running for model", modelstr)
    print("++++++++++++++++++++++++++++++++++++++++++++++")
    MODEL = trainModel(modelstr)
    for tranch in range(1, 4):
        print(f"~~~~~~      tranch {tranch}       ~~~~~~~")
        print()
        img_dir = os.path.join(config.img_dir, f"tranch{tranch}")
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
            
            save_dir = os.path.join(config.save_dir, f"{tranch}/{modelstr}-{n}")
            
            train_model(MODEL.newModel(), 
                        *gens,
                        save_dir = save_dir,
                        epochs = config.EPOCHS 
                        )



if __name__ == '__main__':  
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
     
    train(config.ensemble_num)