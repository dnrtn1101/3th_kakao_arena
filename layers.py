import tensorflow as tf
import tensorflow.keras.backend as K

class DenseTranspose(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 activation=None,
                 kernel_initializer=None,
                 bias_initializer='zeros',
                 tied_to=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.tied_to = tied_to

    def build(self, input_shape):
        self.kernel = K.transpose(self.tied_to.kernel)
        self.non_trainable_weights.append(self.kernel)
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer=self.bias_initializer,
                                    # regularizer=tf.keras.regularizers.l2(1e-10),
                                    name='bias')
        self.built = True

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        output = K.bias_add(output, self.bias, data_format='channels_last')
        output = self.activation(output)

        return output
