import tensorflow as tf
from layers import DenseTranspose

class Encoder(tf.keras.layers.Layer):
    def __init__(self, dim_latent):
        super(Encoder, self).__init__()
        self.latent_layer = tf.keras.layers.Dense(
            units=dim_latent,
            activation='sigmoid',
            kernel_initializer=tf.initializers.GlorotUniform(),
            bias_initializer='zeros')
        self.dropout=tf.keras.layers.Dropout(rate=0.2)

    def call(self, inputs, training=None):
        latent_vector = self.latent_layer(inputs)
        if training:
            latent_vector = self.dropout(latent_vector, training=training)
        return latent_vector

class Decoder(tf.keras.layers.Layer):
    def __init__(self, dim_output, tied_to=None):
        super(Decoder, self).__init__()
        self.tied_to = tied_to
        if self.tied_to is not None:
            self.output_layer = DenseTranspose(units=dim_output,
                                               activation=tf.nn.sigmoid,
                                               kernel_initializer=tf.initializers.GlorotUniform(),
                                               bias_initializer='zeros',
                                               tied_to=self.tied_to)
        else:
            self.output_layer = tf.keras.layers.Dense(
                units=dim_output,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.initializers.GlorotUniform(),
                bias_initializer='zeros'
            )

    def call(self, latent_vectors):
        return self.output_layer(latent_vectors)

class Tied_AutoEncoder(tf.keras.Model):
    def __init__(self, dim_input, dim_latent):
        super(Tied_AutoEncoder, self).__init__()
        self.encoder = Encoder(dim_latent = dim_latent)
        self.tied_to = self.encoder.latent_layer
        self.decoder = Decoder(tied_to=self.tied_to, dim_output = dim_input)

    def call(self, inputs):
        latent_vector = self.encoder(inputs)
        reconstructed = self.decoder(latent_vector)
        return reconstructed

class AutoEncoder(tf.keras.Model):
    def __init__(self, dim_input, dim_latent):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(dim_latent = dim_latent)
        self.decoder = Decoder(dim_output = dim_input)

    def call(self, inputs):
        latent_vector = self.encoder(inputs)
        reconstructed = self.decoder(latent_vector)
        return reconstructed

class textCNN(tf.keras.Model):
    def __init__(self,
                 embedding_matrix,
                 max_len,
                 output_size,
                 num_filter=100,
                 filter_sizes=[3,4,5],
                 drop_prob=0.5,
                 train_embedding=True,
                 **kwargs):
        super(textCNN, self).__init__(**kwargs)
        self.max_len = max_len
        self.output_size = output_size
        self.num_filter = num_filter
        self.drop_prob = drop_prob

        vocab_size, embedding_dim = embedding_matrix.shape

        # layers
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                         output_dim=embedding_dim,
                                                         embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                                         trainable=train_embedding)
        self.conv_layers = []
        for filter_size in filter_sizes:
            conv_block = tf.keras.Sequential()
            conv = tf.keras.layers.Conv2D(filters=num_filter,
                                          kernel_size=(filter_size, embedding_dim),
                                          padding='valid',
                                          activation='relu',
                                          name=f'conv_layer_{filter_size}')
            pooling = tf.keras.layers.MaxPool2D(pool_size=(max_len - filter_size +1, 1),
                                                padding='valid',
                                                strides=(1,1),
                                                name=f'pooling_layer_{filter_size}')
            conv_block.add(conv)
            conv_block.add(pooling)
            self.conv_layers.append(conv_block)

        self.fc = tf.keras.layers.Dense(output_size,
                                        activation='sigmoid',
                                        name='prediction')
        self.reshape = tf.keras.layers.Reshape((max_len, embedding_dim, 1))
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.dropout = tf.keras.layers.Dropout(rate=drop_prob, name='dropout')

    def call(self, inputs, training=None):
        embedding = self.embedding_layer(inputs)
        embedding = self.reshape(embedding)
        output = [layer(embedding) for layer in self.conv_layers]
        output = tf.keras.layers.concatenate(output, axis = -1, name='concatenate')
        output = self.flatten(output)

        if training:
            output=self.dropout(output)

        logit = self.fc(output)
        return logit






