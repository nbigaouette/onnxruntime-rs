import tensorflow as tf
import numpy as np
import tf2onnx


class UniqueModel(tf.keras.Model):

    def __init__(self, name='model1', **kwargs):
        super(UniqueModel, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        return tf.unique(inputs)


model1 = UniqueModel()

print(model1(tf.constant(["foo", "bar", "foo", "baz"])))

model1.save("models/unique_model")
