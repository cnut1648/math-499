import tensorflow as tf


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

    # if stride != 1:
    #     self.downsample = tf.keras.Sequential()
    #     self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
    #                                                kernel_size=(1, 1),
    #                                                strides=stride))
    #     self.downsample.add(tf.keras.layers.BatchNormalization())
    # else:
    #     self.downsample = lambda x: x
    # tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))

    # @tf.function
    def call(self, inputs: list, training=None) -> tuple:
        x, residual = inputs[0], inputs[1]
        # residual = self.downsample(residual)
        x = tf.keras.layers.Concatenate(axis=3)([x, residual])
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        output = tf.nn.relu(x)
        # print("\t with x ",x.shape, "with residual", residual.shape)
        # output = tf.nn.relu(tf.keras.layers.Concatenate(axis=3, name='concat')([x, residual]))
        # residual = tf.keras.layers.Lambda(lambda x: x[...,0:1] if len(x) == 4 else x)(residual)
        # print("\t after output ", output.shape, " with residual", residual.shape)
        # output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return output, residual

    def get_config(self):
        cfg = super().get_config()
        return cfg


class team2net(tf.keras.Model):
    def __init__(self, layer_params):
        super().__init__()
        self.layer_params = layer_params
        self.zeroPadding1 = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            use_bias=False)
        self.convEdge = tf.keras.layers.Conv2D(filters=1,
                                               kernel_size=(7, 7),
                                               strides=2,
                                               use_bias=False,
                                               name='convEdge')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=3)
        self.zeroPadding2 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax)

    def resblock(self, inputs, residual, filter_num, blocks):
        # print(f"resblock with stride = {stride}, ", inputs.shape, residual.shape)
        # inputs, residual = team2netBlock(filter_num, stride=stride)([inputs, residual])
        for i in range(blocks):
            # for i in range(1, blocks):
            # print("for i in range loop (stride = 1) with i = ", i, " shape", inputs.shape, residual.shape)
            inputs, residual = team2netBlock(filter_num, stride=1)([inputs, residual])
            # print()
        # print()
        return inputs, residual

    def transition_block(self, input, residual, reduction):
        x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5)(input)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Conv2D(int(x.shape[3] * reduction), 1)(x)
        avgpool = tf.keras.layers.AveragePooling2D(2, strides=2)
        return avgpool(x), avgpool(residual)

    def call(self, inputs, training=None):
        x = tf.keras.layers.Lambda(lambda x: x[..., :3])(inputs)
        residual = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[..., -1], -1))(inputs)

        print(x)
        x = self.zeroPadding1(x)
        residual = self.zeroPadding1(residual)
        print(x)

        x = self.conv1(x)
        residual = self.convEdge(residual)
        print(x.shape)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.zeroPadding2(x)
        residual = self.zeroPadding2(residual)
        x = self.pool1(x)
        residual = self.pool1(residual)
        print("after 1st ,", x.shape, residual.shape)
        # start skip
        x, residual = self.resblock(x, residual, filter_num=64,
                                    blocks=self.layer_params[0])
        print("after 1st resblock,", x.shape, residual.shape)
        x, residual = self.transition_block(input=x, residual=residual, reduction=0.5)
        print("after 1st transition,", x.shape, residual.shape)

        x, residual = self.resblock(x, residual, filter_num=128,
                                    blocks=self.layer_params[1])
        print("after 2nd resblock,", x.shape, residual.shape)

        x, residual = self.resblock(x, residual, filter_num=256,
                                    blocks=self.layer_params[2])

        x, residual = self.resblock(x, residual, filter_num=512,
                                    blocks=self.layer_params[3])

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
