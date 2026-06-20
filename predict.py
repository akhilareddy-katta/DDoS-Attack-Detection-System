import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("ddos_model.h5")

def detect(sample):
    sample = np.array(sample).reshape(1,-1)
    pred = model.predict(sample)

    if pred > 0.5:
        return "DDoS Attack"
    else:
        return "Normal Traffic"