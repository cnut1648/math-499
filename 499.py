from zipfile import ZipFile
import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A
from tensorflow.keras import datasets, layers, models
import numpy as np
import seaborn as sns
import pandas as pd

"""# team2net"""

merge_dir = "/content/drive/My Drive/M499/merge3/"
tranch = 3
labels_path = '/content/drive/My Drive/M499/tranch'+str(tranch)+'_labels.csv'
pictures_path = '/content/drive/My Drive/M499/persons-posture-tranch'+str(tranch)+'.zip'

labels = pd.read_csv(labels_path)
zip_file = ZipFile(pictures_path)

file_list = [obj.filename for obj in zip_file.infolist()]
file_list_simple = [name.split('/')[-1] for name in file_list]

df = labels
len(df)

df["final_url"] = df["final_url"].str.replace("PNG", 'png')
df["final_url"] = df["final_url"].str.replace("JPG", 'png')

df = df.dropna()
df.index = range(len(df))
df["primary_posture"].value_counts()

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.33, random_state=42)
train = train.query("primary_posture != 'Unknown'")
test = test.query("primary_posture != 'Unknown'")
len(train), len(test)

train["primary_posture"] .value_counts() / len(train)

test["primary_posture"] .value_counts()  / len(test)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(data_format="channels_last",
                            #  rescale = 1./255,
                            shear_range=0.1,
                            zoom_range=0.1,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            brightness_range=[0.9,1.1],
                            horizontal_flip=True,
                            validation_split=0.2,
                            # preprocessing_function=custom_preprocess
)

val_datagen = ImageDataGenerator(data_format="channels_last",
                            #  rescale = 1./255,
                             validation_split=0.2,
                            #  preprocessing_function= mobilenet_v2.preprocess_input
)

test_datagen = ImageDataGenerator(data_format="channels_last",
                            #  rescale = 1./255,
                            #  preprocessing_function= mobilenet_v2.preprocess_input
)

train_gen = train_datagen.flow_from_dataframe(train,
                            directory = "/content/drive/My Drive/M499/merge3/",
                            x_col = "final_url",
                            y_col = "primary_posture",
                            class_mode="sparse",
                            color_mode="rgba",
                            batch_size = 32,
                            seed=42,
                            subset = "training",
                            shuffle=True,
                            target_size=(224, 224),
                            validate_filenames = True,
)

val_gen = val_datagen.flow_from_dataframe(train,
                            directory = "/content/drive/My Drive/M499/merge3/",
                            x_col = "final_url",
                            y_col = "primary_posture",
                            class_mode="sparse",
                            color_mode="rgba",
                            batch_size = 32,
                            seed=42,
                            subset = "validation",
                            shuffle=True,
                            target_size=(224, 224),
                            validate_filenames = True,
)


test_gen = test_datagen.flow_from_dataframe(test,
                            directory = "/content/drive/My Drive/M499/merge3/",
                            x_col = "final_url",
                            y_col = "primary_posture",
                            class_mode="sparse",
                            color_mode="rgba",
                            batch_size = 32,
                            seed=42,
                            shuffle=True,
                            target_size=(224, 224),
                            validate_filenames = True,
)

import tensorflow as tf
import numpy as np


class team2netBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same', name='block_conv1', input_shape=(56, 56, 3))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding='same', name='block_conv2', input_shape=(56, 56, filter_num))
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    # @tf.function
    def call(self, inputs: list, training=None):
        x, residual = inputs[0], inputs[1]
        residual = self.downsample(residual)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return output, residual


class team2net(tf.keras.Model):
    def __init__(self, layer_params):
        super().__init__()
        self.layer_params = layer_params
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.convEdge = tf.keras.layers.Conv2D(filters=1,
                                               kernel_size=(7, 7),
                                               strides=2,
                                               padding="same", name='convEdge')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax)

    def resblock(self, inputs, residual, filter_num, blocks, stride):
        inputs, residual = team2netBlock(filter_num, stride=stride)([inputs, residual])
        for i in range(1, blocks):
            inputs, residual = team2netBlock(filter_num, stride=1)([inputs, residual])
        return inputs, residual


    def call(self, inputs, training=None):
        x = tf.keras.layers.Lambda(lambda x: x[..., :3])(inputs)
        residual = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[..., -1], -1))(inputs)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        residual = self.convEdge(residual)
        residual = self.pool1(residual)

        # start skip
        x, residual = self.resblock(x, residual, filter_num=64,
                                    blocks = self.layer_params[0], stride =1)

        x, residual = self.resblock(x, residual, filter_num=128,
                                    blocks = self.layer_params[1], stride =2)

        x, residual = self.resblock(x, residual, filter_num=256,
                                    blocks = self.layer_params[2], stride =2)

        x, residual = self.resblock(x, residual, filter_num=512,
                                    blocks = self.layer_params[3], stride =2)

        x = self.avgpool(x)
        output = self.fc(x)
        return output

    def model(self):
        x = tf.keras.Input(shape=(224, 224, 4))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


team2net34 = team2net([3, 4, 6, 3]).model()

from tensorflow.keras import optimizers, losses
team2net34.build(input_shape=(None, 224, 224, 4))
team2net34.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy'])
team2net34.summary()

def train_model(model):
  from datetime import datetime
  import json
  
  save_dir = "/content/drive/My Drive/model/team2net"
 
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
                                                  patience = 13,
                                                  verbose = 1,
                                                  restore_best_weights = True)
  

  txt_log = open(save_dir+".log", mode='wt', buffering=1)
  
  log_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end = lambda epoch, logs: txt_log.write(
      json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
    on_train_end = lambda logs: txt_log.close()
  )


  his = model.fit(train_gen, 
      steps_per_epoch=171,
      epochs=60,
      callbacks=[
                 tf.keras.callbacks.ModelCheckpoint(filepath=save_dir + '.{epoch:02d}-{val_accuracy:.2f}.h5'),
                 early_stopper, reduce_lr,
                 log_callback,
                 ],
      verbose = 1,
      # class_weight = class_weights,
      validation_data = val_gen,
      validation_steps = 42,
      shuffle = True,
  )
  plt.plot(his.history["accuracy"], label = "training")
  plt.plot(his.history["val_accuracy"], label = "validating")
  plt.legend()
  plt.savefig(save_dir + "plot.jpg")

  cur = datetime.now()
  save_dir += str(cur.day) + "-" + str(cur.hour)

  model.save(save_dir)
train_model(team2net34)
