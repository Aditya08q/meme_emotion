import tensorflow as tf
import argparse
import os

def convert(keras_path, out_path, quantize=False):
    model = tf.keras.models.load_model(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print("Saved tflite:", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--keras', default='model/emotion_model.h5')
    parser.add_argument('--out', default='model/emotion_model.tflite')
    parser.add_argument('--quantize', action='store_true')
    args = parser.parse_args()
    convert(args.keras, args.out, args.quantize)
