import tensorflow as tf
import numpy as np
import tf2onnx


class RegexModel(tf.keras.Model):

    def __init__(self, name='model1', **kwargs):
        super(RegexModel, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        return tf.strings.regex_replace(inputs, " ", "_", replace_global=True)


model1 = RegexModel()

print(model1(tf.constant(["Hello world!"])))

model1.save("models/regex_model")
